//! Save slots in the browser's IndexedDB (a scene can be megabytes: too big for
//! localStorage), and scene files to download or open.

use js_sys::{Promise, Uint8Array};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{IdbDatabase, IdbRequest, IdbTransactionMode};

const DATABASE: &str = "embloks";
const STORE: &str = "slots";

/// A promise settled by an IndexedDB request's success or error.
fn settled(request: &IdbRequest) -> JsFuture {
    let request = request.clone();
    JsFuture::from(Promise::new(&mut |resolve, reject| {
        let ok = request.clone();
        let on_success = Closure::once_into_js(move |_: web_sys::Event| {
            let _ = resolve.call1(&JsValue::NULL, &ok.result().unwrap_or(JsValue::UNDEFINED));
        });
        let on_error = Closure::once_into_js(move |_: web_sys::Event| {
            let _ = reject.call1(&JsValue::NULL, &JsValue::from_str("IndexedDB request failed"));
        });
        request.set_onsuccess(Some(on_success.unchecked_ref()));
        request.set_onerror(Some(on_error.unchecked_ref()));
    }))
}

async fn open() -> Result<IdbDatabase, JsValue> {
    let factory = web_sys::window()
        .and_then(|w| w.indexed_db().ok().flatten())
        .ok_or_else(|| JsValue::from_str("IndexedDB is not available"))?;
    let request = factory.open_with_u32(DATABASE, 1)?;
    let upgrade = Closure::<dyn FnMut(web_sys::Event)>::new(move |event: web_sys::Event| {
        let request: web_sys::IdbOpenDbRequest = event.target().unwrap().unchecked_into();
        let db: IdbDatabase = request.result().unwrap().unchecked_into();
        if !db.object_store_names().contains(STORE) {
            let _ = db.create_object_store(STORE);
        }
    });
    request.set_onupgradeneeded(Some(upgrade.as_ref().unchecked_ref()));
    let db = settled(&request).await?;
    drop(upgrade);
    Ok(db.unchecked_into())
}

fn key(slot: u8) -> JsValue {
    JsValue::from_str(&format!("slot{slot}"))
}

pub async fn put(slot: u8, bytes: &[u8]) -> Result<(), String> {
    let run = async {
        let db = open().await?;
        let store = db.transaction_with_str_and_mode(STORE, IdbTransactionMode::Readwrite)?.object_store(STORE)?;
        let value = Uint8Array::from(bytes);
        settled(&store.put_with_key(&value, &key(slot))?).await?;
        Ok::<(), JsValue>(())
    };
    run.await.map_err(describe)
}

pub async fn get(slot: u8) -> Result<Option<Vec<u8>>, String> {
    let run = async {
        let db = open().await?;
        let store = db.transaction_with_str(STORE)?.object_store(STORE)?;
        let value = settled(&store.get(&key(slot))?).await?;
        Ok::<_, JsValue>(if value.is_undefined() { None } else { Some(Uint8Array::new(&value).to_vec()) })
    };
    run.await.map_err(describe)
}

fn describe(error: JsValue) -> String {
    error.as_string().unwrap_or_else(|| format!("{error:?}"))
}

/// Offer `bytes` as a file download.
pub fn download(bytes: &[u8], name: &str) -> Result<(), JsValue> {
    let parts = js_sys::Array::of1(&Uint8Array::from(bytes));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("application/octet-stream");
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;
    let document = web_sys::window().unwrap().document().unwrap();
    let link: web_sys::HtmlAnchorElement = document.create_element("a")?.unchecked_into();
    link.set_href(&url);
    link.set_download(name);
    link.click();
    web_sys::Url::revoke_object_url(&url)
}

/// The contents of the file picked in an <input type=file>.
pub async fn read_file(file: web_sys::File) -> Result<Vec<u8>, String> {
    let buffer = JsFuture::from(file.array_buffer()).await.map_err(describe)?;
    Ok(Uint8Array::new(&buffer).to_vec())
}
