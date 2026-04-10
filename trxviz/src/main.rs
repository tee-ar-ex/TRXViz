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

    let trx_path = std::env::args().nth(1);
    let nifti_path = std::env::args().nth(2);

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
        Box::new(move |cc| Ok(Box::new(app::TrxVizApp::new(cc, trx_path, nifti_path)))),
    )
}
