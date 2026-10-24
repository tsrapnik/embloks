//! The HTML panels drawn over the canvas: info panel, layers panel and toolbar.
//! Each is rendered to an HTML string from the app's state; clicks are delegated
//! through `data-action` attributes (see app.rs).

use embloks_core::brush::Tool;
use embloks_core::controller::{App, HudSection, HudValue, Key, Mods};
use embloks_core::format::{describe_frequency, is_number};
use embloks_core::layers::{LayerId, Layers};
use embloks_core::scenes::SCENES;

pub fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// The info panel: sections of (label, value) rows. Tuple values become fixed columns
/// with numbers right-aligned in a monospace font, so digits stay put.
pub fn info_html(sections: &[HudSection]) -> String {
    let mut html = String::new();
    for section in sections.iter().filter(|s| !s.is_empty()) {
        html += "<section>";
        for (label, value) in section {
            html += &format!("<div class=\"label\">{}</div>", escape(label));
            match value {
                HudValue::Text(text) => html += &format!("<div class=\"value\">{}</div>", escape(text)),
                HudValue::Fields(fields) => {
                    html += "<div class=\"value fields\">";
                    for field in fields {
                        let class = if is_number(field) { "num" } else { "" };
                        html += &format!("<span class=\"{class}\">{}</span>", escape(field));
                    }
                    html += "</div>";
                }
            }
        }
        html += "</section>";
    }
    html
}

fn layer_key(id: LayerId) -> &'static str {
    match id {
        LayerId::E => "E",
        LayerId::B => "B",
        LayerId::S => "S",
        LayerId::P => "P",
    }
}

pub fn layer_id(key: &str) -> Option<LayerId> {
    LayerId::ALL.into_iter().find(|&id| layer_key(id) == key)
}

/// The layers panel: per layer show, scale (-, value or auto, +), opacity, log/linear.
pub fn layers_html(layers: &Layers) -> String {
    let button = |id: LayerId, control: &str, text: &str, on: bool, class: &str| {
        format!(
            "<button class=\"{class}{}\" data-layer=\"{}\" data-control=\"{control}\">{text}</button>",
            if on { " on" } else { "" },
            layer_key(id)
        )
    };
    let mut html = String::from(
        "<table><tr><th>Layers</th><th>show</th><th colspan=\"3\">scale</th><th colspan=\"3\">opacity</th><th>mode</th></tr>",
    );
    for id in LayerId::ALL {
        let layer = &layers[id];
        html += &format!("<tr><td>{}</td><td>{}</td>", escape(&layer.label), button(id, "show", if layer.visible { "on" } else { "off" }, layer.visible, ""));
        if layer.adjustable {
            html += &format!(
                "<td>{}</td><td>{}</td><td>{}</td>",
                button(id, "scale-", "−", false, "step"),
                button(id, "auto", &escape(&layer.gain_label()), layer.auto, "value"),
                button(id, "scale+", "+", false, "step"),
            );
            html += &format!(
                "<td>{}</td><td><span class=\"value\">{:.0}%</span></td><td>{}</td>",
                button(id, "opacity-", "−", false, "step"),
                layer.opacity * 100.0,
                button(id, "opacity+", "+", false, "step"),
            );
            html += &format!("<td>{}</td>", button(id, "mode", if layer.log_scale { "log" } else { "linear" }, false, ""));
        } else {
            html += "<td colspan=\"7\"></td>";
        }
        html += "</tr>";
    }
    html + "</table>"
}

/// The toolbar: tool and its value, brush, simulation speed, scenes, saves.
pub fn toolbar_html(app: &App, erase_mode: bool, slot: u8) -> String {
    let b = &app.brush;
    let action = |name: &str, text: &str, on: bool, title: &str| {
        format!(
            "<button class=\"{}\" data-action=\"{name}\" title=\"{title}\">{text}</button>",
            if on { "on" } else { "" }
        )
    };
    let step = |name: &str, text: &str, title: &str| format!("<button class=\"step\" data-action=\"{name}\" title=\"{title}\">{text}</button>");
    let value = match b.tool {
        Tool::Conductor => format!("{} Ω", b.resistance()),
        Tool::Battery => format!("EMF {}", b.emf()),
        Tool::Charge => format!("{:+} e", b.charge()),
    };
    let mut html = String::new();
    html += "<div class=\"group\">";
    html += &action("tool-1", "Wire", b.tool == Tool::Conductor, "Conductor (1)");
    html += &action("tool-2", "Battery", b.tool == Tool::Battery, "Battery (2)");
    html += &action("tool-3", "Charge", b.tool == Tool::Charge, "Test charge (3)");
    html += &action("erase", "Erase", erase_mode, "Paint erases (like right-dragging)");
    html += "</div><div class=\"group\">";
    html += &step("value-", "−", "Smaller value (Down)");
    html += &format!("<span class=\"readout\">{}</span>", escape(&value));
    html += &step("value+", "+", "Larger value (Up)");
    if b.tool == Tool::Battery {
        html += &action("rotate", &format!("+ {}", b.orientation.name()), false, "Rotate the battery (R)");
        html += &action("frequency", &describe_frequency(b.frequency()), b.frequency() > 0.0, "DC or AC (F)");
    }
    html += "</div><div class=\"group\"><span class=\"caption\">brush</span>";
    html += &step("brush-", "−", "Smaller brush ([)");
    html += &format!("<span class=\"readout\">{}</span>", b.radius);
    html += &step("brush+", "+", "Larger brush (])");
    html += "</div><div class=\"group\">";
    html += &action("pause", if app.paused { "▶" } else { "❚❚" }, app.paused, "Pause / resume (Space)");
    html += &step("speed-", "«", "Slower (,)");
    html += &format!("<span class=\"readout\">{}×</span>", app.sim_speed);
    html += &step("speed+", "»", "Faster (.)");
    html += "</div><div class=\"group\"><select data-action=\"scene\" title=\"Example scenes (F1-F7)\"><option value=\"\">Examples…</option>";
    for (i, scene) in SCENES.iter().enumerate() {
        html += &format!("<option value=\"{i}\">F{} {}</option>", i + 1, escape(scene.title));
    }
    html += "</select>";
    html += &action("clear", "Clear", false, "Clear the canvas (C)");
    html += &step("density-", "−", "Coarser grid (-), clears");
    html += &format!("<span class=\"readout\">{}×{}</span>", app.grid.cols, app.grid.rows);
    html += &step("density+", "+", "Finer grid (+), clears");
    html += "</div><div class=\"group\"><select data-action=\"slot\" title=\"Save slot\">";
    for s in 0..10 {
        html += &format!("<option value=\"{s}\"{}>slot {s}</option>", if s == slot { " selected" } else { "" });
    }
    html += "</select>";
    html += &action("save", "Save", false, "Save to the slot (Ctrl+Shift+digit)");
    html += &action("load", "Load", false, "Load the slot (Shift+digit)");
    html += &action("export", "Export", false, "Download the scene as a file");
    html += &action("import", "Import", false, "Open a scene file");
    html += &action("help", "Info", false, "Info panel: full / status / hidden (H)");
    html + "</div>"
}

/// A key event as the app understands it, if it is one of its keys.
pub fn map_key(code: &str, key: &str, ctrl: bool, shift: bool) -> Option<(Key, Mods)> {
    let mods = Mods { ctrl, shift };
    // Digits by physical key, so Shift+1 is still 1 (not "!") on any layout.
    if let Some(n) = code.strip_prefix("Digit").or_else(|| code.strip_prefix("Numpad")).and_then(|d| d.parse::<u8>().ok()) {
        return Some((Key::Digit(n), mods));
    }
    if let Some(n) = code.strip_prefix('F').and_then(|n| n.parse::<u8>().ok()) {
        return Some((Key::Function(n), mods));
    }
    let key = match key {
        "ArrowUp" => Key::Up,
        "ArrowDown" => Key::Down,
        " " => Key::Space,
        "Escape" => Key::Escape,
        k if k.chars().count() == 1 => Key::Char(k.chars().next()?.to_ascii_lowercase()),
        _ => return None,
    };
    // Leave the browser's own shortcuts (Ctrl+R, Ctrl+C, ...) alone.
    if ctrl && matches!(key, Key::Char(_)) {
        return None;
    }
    Some((key, mods))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_map_like_the_desktop_app() {
        assert_eq!(map_key("Digit3", "#", false, true), Some((Key::Digit(3), Mods { ctrl: false, shift: true })));
        assert_eq!(map_key("F7", "F7", false, false).unwrap().0, Key::Function(7));
        assert_eq!(map_key("Comma", ",", false, false).unwrap().0, Key::Char(','));
        assert_eq!(map_key("KeyR", "R", false, true).unwrap().0, Key::Char('r'));
        assert_eq!(map_key("Space", " ", false, false).unwrap().0, Key::Space);
        assert_eq!(map_key("KeyC", "c", true, false), None);
        assert_eq!(map_key("ShiftLeft", "Shift", false, true), None);
    }

    #[test]
    fn info_panel_marks_numbers_and_escapes_text() {
        let sections = vec![vec![
            ("V".to_string(), HudValue::Fields(vec!["+0.5000".into(), "|E|".into()])),
            ("»".to_string(), HudValue::Text("a < b".into())),
        ]];
        let html = info_html(&sections);
        assert!(html.contains("<span class=\"num\">+0.5000</span>"));
        assert!(html.contains("<span class=\"\">|E|</span>"));
        assert!(html.contains("a &lt; b"));
    }
}
