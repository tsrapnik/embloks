//! Start-up, DOM events and the animation loop.

use std::cell::RefCell;
use std::rc::Rc;

use embloks_core::controller::App;
use embloks_core::settings::Settings;
use embloks_core::layers::LayerControl;
use embloks_gpu::Gpu;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

use crate::app::{Dom, Input, Task, WebApp, button_of, target_element};
use crate::{dom, saves};

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    wasm_bindgen_futures::spawn_local(async {
        if let Err(message) = run().await {
            notice(&message);
        }
    });
}

/// Replace the page with a message (no WebGPU, or something failed at start-up).
fn notice(message: &str) {
    let document = web_sys::window().unwrap().document().unwrap();
    if let Some(element) = document.get_element_by_id("notice") {
        element.set_text_content(Some(message));
        element.remove_attribute("hidden").ok();
    }
}

fn element<T: JsCast>(document: &web_sys::Document, id: &str) -> Result<T, String> {
    document
        .get_element_by_id(id)
        .and_then(|e| e.dyn_into::<T>().ok())
        .ok_or_else(|| format!("page is missing #{id}"))
}

async fn run() -> Result<(), String> {
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let canvas: web_sys::HtmlCanvasElement = element(&document, "canvas")?;
    let dom = Dom {
        info: element(&document, "info")?,
        layers: element(&document, "layers")?,
        toolbar: element(&document, "toolbar")?,
        import: element(&document, "import")?,
        canvas: canvas.clone(),
    };

    // WebGPU only: without it the simulation has nowhere to run.
    let has_webgpu = js_sys::Reflect::get(&window.navigator(), &"gpu".into()).is_ok_and(|gpu| !gpu.is_undefined());
    if !has_webgpu {
        return Err("Embloks needs WebGPU, which this browser doesn't offer. Try a recent Chrome, Edge, \
                    Safari or Firefox (and, on Linux, check that WebGPU is enabled)."
            .into());
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let search = window.location().search().unwrap_or_default();
    let surface = if search.contains("offscreen") {
        None
    } else {
        Some(
            instance
                .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
                .map_err(|e| format!("Could not use the canvas for WebGPU: {e}"))?,
        )
    };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface.as_ref(),
            ..Default::default()
        })
        .await
        .map_err(|e| format!("No WebGPU adapter: {e}"))?;
    let gpu = Gpu::from_adapter(&adapter).await?;

    let ratio = window.device_pixel_ratio();
    let width = ((canvas.client_width() as f64 * ratio) as u32).max(1);
    let height = ((canvas.client_height() as f64 * ratio) as u32).max(1);
    canvas.set_width(width);
    canvas.set_height(height);
    let config = match &surface {
        Some(surface) => {
            let mut config = surface.get_default_config(&adapter, width, height).ok_or("The canvas can't show WebGPU output")?;
            // A non-sRGB format: the colours are given as they should appear.
            if let Some(format) = surface.get_capabilities(&adapter).formats.iter().copied().find(|f| !f.is_srgb()) {
                config.format = format;
            }
            config.present_mode = wgpu::PresentMode::Fifo;
            surface.configure(&gpu.device, &config);
            config
        }
        None => wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            color_space: Default::default(),
        },
    };
    let cpu = search.contains("backend=cpu");
    let seed = (js_sys::Math::random() * u32::MAX as f64) as u64;
    let clock = embloks_core::controller::system_clock();
    let mut app = App::with_clock(Settings::default(), width as i32, height as i32, clock, seed);
    app.load_scene(0);
    // ?paused starts with the physics stopped (handy on slow software GPUs).
    app.paused = search.contains("paused");
    let web = Rc::new(RefCell::new(WebApp::new(app, (instance, adapter), gpu, surface, config, dom, cpu, ratio)));
    listen(&web, &document)?;
    APP.with(|app| *app.borrow_mut() = Some(web.clone()));
    animate(web);
    Ok(())
}

thread_local! {
    static APP: RefCell<Option<Rc<RefCell<WebApp>>>> = const { RefCell::new(None) };
}

/// Ask for a copy of the next frame drawn offscreen (?offscreen); for automated tests.
#[wasm_bindgen]
pub fn request_snapshot() {
    APP.with(|app| {
        if let Some(web) = app.borrow().as_ref() {
            web.borrow_mut().request_snapshot();
        }
    });
}

/// The last requested frame as RGBA rows of the canvas's size, once it arrived.
#[wasm_bindgen]
pub fn snapshot() -> Option<js_sys::Uint8Array> {
    APP.with(|app| app.borrow().as_ref()?.borrow().snapshot().map(js_sys::Uint8Array::from))
}

fn add<E: JsCast + wasm_bindgen::convert::FromWasmAbi + 'static>(target: &web_sys::EventTarget, kind: &str, passive: bool, handler: impl FnMut(E) + 'static) {
    let closure = Closure::<dyn FnMut(E)>::new(handler);
    let options = web_sys::AddEventListenerOptions::new();
    options.set_passive(passive);
    target
        .add_event_listener_with_callback_and_add_event_listener_options(kind, closure.as_ref().unchecked_ref(), &options)
        .ok();
    closure.forget();
}

fn listen(web: &Rc<RefCell<WebApp>>, document: &web_sys::Document) -> Result<(), String> {
    let window: web_sys::EventTarget = web_sys::window().unwrap().into();
    let canvas: web_sys::EventTarget = web.borrow().dom.canvas.clone().into();
    let push = |web: &Rc<RefCell<WebApp>>, input: Input| web.borrow_mut().queue.push(input);
    let position = |web: &Rc<RefCell<WebApp>>, e: &web_sys::PointerEvent| {
        let ratio = web.borrow().pixel_ratio;
        (e.offset_x() as f64 * ratio, e.offset_y() as f64 * ratio)
    };

    let w = web.clone();
    add(&canvas, "pointerdown", false, move |e: web_sys::PointerEvent| {
        e.prevent_default();
        let target: web_sys::Element = w.borrow().dom.canvas.clone().into();
        target.set_pointer_capture(e.pointer_id()).ok();
        let _ = w.borrow().dom.canvas.focus();
        let (x, y) = position(&w, &e);
        push(&w, Input::Move(x, y));
        push(&w, Input::Down(button_of(&e), x, y));
    });
    let w = web.clone();
    add(&canvas, "pointermove", true, move |e: web_sys::PointerEvent| {
        let (x, y) = position(&w, &e);
        push(&w, Input::Move(x, y));
    });
    let w = web.clone();
    add(&canvas, "pointerup", true, move |e: web_sys::PointerEvent| push(&w, Input::Up(button_of(&e))));
    let w = web.clone();
    add(&canvas, "pointercancel", true, move |e: web_sys::PointerEvent| push(&w, Input::Up(button_of(&e))));
    let w = web.clone();
    add(&canvas, "pointerleave", true, move |_: web_sys::PointerEvent| push(&w, Input::Leave));
    add(&canvas, "contextmenu", false, |e: web_sys::Event| e.prevent_default());
    let w = web.clone();
    add(&canvas, "wheel", false, move |e: web_sys::WheelEvent| {
        e.prevent_default();
        let steps = if e.delta_y() < 0.0 { 1 } else { -1 };
        push(&w, Input::Wheel(steps));
    });

    let w = web.clone();
    add(&window, "keydown", false, move |e: web_sys::KeyboardEvent| {
        // Typing into the toolbar's controls is theirs.
        if let Some(target) = target_element(&e)
            && matches!(target.tag_name().as_str(), "INPUT" | "SELECT" | "TEXTAREA")
        {
            return;
        }
        if let Some((key, mods)) = dom::map_key(&e.code(), &e.key(), e.ctrl_key() || e.meta_key(), e.shift_key()) {
            e.prevent_default();
            push(&w, Input::Key(key, mods));
        }
    });

    // Layers panel and toolbar: delegated clicks on their buttons.
    let w = web.clone();
    let layers: web_sys::EventTarget = web.borrow().dom.layers.clone().into();
    add(&layers, "click", true, move |e: web_sys::MouseEvent| {
        let Some(button) = target_element(&e).and_then(|t| t.closest("button").ok().flatten()) else { return };
        let (Some(layer), Some(control)) = (button.get_attribute("data-layer"), button.get_attribute("data-control")) else { return };
        let control = match control.as_str() {
            "show" => LayerControl::Show,
            "scale-" => LayerControl::ScaleDown,
            "scale+" => LayerControl::ScaleUp,
            "auto" => LayerControl::ScaleAuto,
            "opacity-" => LayerControl::OpacityDown,
            "opacity+" => LayerControl::OpacityUp,
            _ => LayerControl::Mode,
        };
        if let Some(id) = dom::layer_id(&layer) {
            push(&w, Input::Layer(id, control));
        }
    });
    let toolbar: web_sys::EventTarget = web.borrow().dom.toolbar.clone().into();
    let w = web.clone();
    add(&toolbar, "click", true, move |e: web_sys::MouseEvent| {
        let Some(button) = target_element(&e).and_then(|t| t.closest("button").ok().flatten()) else { return };
        if let Some(action) = button.get_attribute("data-action") {
            push(&w, Input::Action(action, String::new()));
        }
    });
    let w = web.clone();
    add(&toolbar, "change", true, move |e: web_sys::Event| {
        let Some(select) = target_element(&e).and_then(|t| t.dyn_into::<web_sys::HtmlSelectElement>().ok()) else { return };
        if let Some(action) = select.get_attribute("data-action") {
            push(&w, Input::Action(action, select.value()));
        }
        // Back to the canvas, so keys go to the app again.
        let _ = w.borrow().dom.canvas.focus();
    });

    // Opening a scene file.
    let w = web.clone();
    let import: web_sys::EventTarget = web.borrow().dom.import.clone().into();
    add(&import, "change", true, move |_: web_sys::Event| {
        let input = w.borrow().dom.import.clone();
        let Some(file) = input.files().and_then(|files| files.get(0)) else { return };
        input.set_value("");
        let w = w.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = saves::read_file(file).await;
            w.borrow_mut().tasks.push(Task::Imported(result));
        });
    });
    let _ = document;
    Ok(())
}

/// Run a frame on every animation frame.
fn animate(web: Rc<RefCell<WebApp>>) {
    type FrameCallback = Closure<dyn FnMut(f64)>;
    let callback: Rc<RefCell<Option<FrameCallback>>> = Rc::new(RefCell::new(None));
    let next = callback.clone();
    *callback.borrow_mut() = Some(Closure::new(move |now: f64| {
        web.borrow_mut().frame(&web, now);
        request_frame(next.borrow().as_ref().unwrap());
    }));
    request_frame(callback.borrow().as_ref().unwrap());
}

fn request_frame(callback: &Closure<dyn FnMut(f64)>) {
    web_sys::window().unwrap().request_animation_frame(callback.as_ref().unchecked_ref()).ok();
}
