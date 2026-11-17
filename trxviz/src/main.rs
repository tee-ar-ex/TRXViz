mod app;

fn load_icon() -> egui::IconData {
    let png_bytes = include_bytes!("../assets/logo-512x512.png");
    let image = image::load_from_memory(png_bytes)
        .expect("Failed to load embedded icon")
        .into_rgba8();
    let (width, height) = image.dimensions();
    egui::IconData {
        rgba: image.into_raw(),
        width,
        height,
    }
}

fn main() -> eframe::Result {
    env_logger::init();

    let mut startup_paths: Vec<std::path::PathBuf> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--workflow" || arg == "-w" {
            continue;
        } else if arg == "--python-env" || arg == "--mpat-python" {
            if let Some(py_path) = args.next() {
                trxviz_core::workflow::set_cli_python_path(py_path);
            }
        } else if let Some(py_path) = arg
            .strip_prefix("--python-env=")
            .or_else(|| arg.strip_prefix("--mpat-python="))
        {
            trxviz_core::workflow::set_cli_python_path(py_path.to_string());
        } else {
            startup_paths.push(std::path::PathBuf::from(arg));
        }
    }

    let icon = load_icon();

    let wgpu_options = egui_wgpu::WgpuConfiguration {
        wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew {
            device_descriptor: std::sync::Arc::new(|adapter| {
                let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                };
                wgpu::DeviceDescriptor {
                    label: Some("trxviz device"),
                    required_limits: wgpu::Limits {
                        max_texture_dimension_2d: 8192,
                        max_buffer_size: 1 << 30, // 1 GB
                        ..base_limits
                    },
                    ..Default::default()
                }
            }),
            ..Default::default()
        }),
        ..Default::default()
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("TRXViz")
            .with_inner_size([1280.0, 800.0])
            .with_icon(icon),
        renderer: eframe::Renderer::Wgpu,
        depth_buffer: 32,
        wgpu_options,
        ..Default::default()
    };

    eframe::run_native(
        "TRXViz",
        options,
        Box::new(move |cc| Ok(Box::new(app::TrxVizApp::new(cc, startup_paths)))),
    )
}
