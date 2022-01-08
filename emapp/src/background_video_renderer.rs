use crate::{uri::Uri, error::Exception};

pub trait BackgroundVideoRenderer<Error: Exception> {
    fn load(&mut self, file_uri: &Uri) -> Result<(), Error>;
    // TODO: use sokol or other
    fn draw(&mut self);
    fn seek(&mut self, sec: f64);
    fn flush(&mut self);
    fn destroy(&mut self);
    fn file_uri<'a: 'b, 'b>(&'a self) -> &'b Uri;
}