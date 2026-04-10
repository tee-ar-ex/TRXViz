impl super::super::TrxVizApp {
    pub(in crate::app) fn show_advanced_shell(
        &mut self,
        ctx: &egui::Context,
        frame: &mut eframe::Frame,
    ) {
        self.show_workspace(ctx, frame);
    }
}
