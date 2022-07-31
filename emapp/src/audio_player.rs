use crate::error::Error;

pub struct Rational {
    pub numerator: u64,
    pub denominator: u32,
}


pub trait AudioPlayer {
    // TODO
    fn load(&mut self, bytes: &[u8]) -> Result<(), Error>;
    fn play(&mut self);
    fn play_part(&mut self, start: f64, length: f64);
    fn pause(&mut self);
    fn resume(&mut self);
    fn suspend(&mut self);
    fn stop(&mut self);
    fn update(&mut self);
    fn seek(&mut self, rational: Rational);

    /// Verify bias between current rational and expect rational is small. 
    /// If diff between current and expect larger than epsilon, player will seek to expect. 
    /// 
    /// # Return
    /// 
    /// Return `Ok(current_rational)` if less than epsilon,
    /// Return `Err(prev_rational)` else. 
    fn verify_bias(&mut self, expect: Rational, epsilon: Rational) -> Result<Rational, Rational>;

    fn current_rational(&self) -> Rational;
    fn last_rational(&self) -> Rational;
    fn duration_rational(&self) -> Rational;
    fn bits_per_sample(&self) -> u32;
    fn num_channels(&self) -> u32;
    fn sample_rate(&self) -> u32;

    fn is_loaded(&self) -> bool;
    fn is_finished(&self) -> bool;
    fn is_playing(&self) -> bool;
    fn is_paused(&self) -> bool;
    fn is_stopped(&self) -> bool;
    fn was_playing(&self) -> bool;

}