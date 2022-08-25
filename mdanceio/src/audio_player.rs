use std::time::{Duration, Instant};

use crate::error::MdanceioError;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Rational {
    pub numerator: u64,
    pub denominator: u32,
}

impl Rational {
    pub fn new(numerator: u64, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    pub fn subdivide(&self) -> f64 {
        self.numerator as f64 / self.denominator.max(1) as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPlayerState {
    Initial,
    Stop,
    Suspend,
    Pause,
    Play,
}

pub trait AudioPlayer {
    // TODO
    fn load(bytes: &[u8]) -> Result<Self, MdanceioError>
    where
        Self: Sized;
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
    fn last_rational(&self) -> Option<Rational>;
    fn duration_rational(&self) -> Option<Rational>;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockState {
    Stopping,
    Running(Instant),
    Pausing(Duration),
}

pub struct Clock {
    state: ClockState,
}

impl Clock {
    pub fn new() -> Self {
        Self {
            state: ClockState::Stopping,
        }
    }

    pub fn is_paused(&self) -> bool {
        matches!(self.state, ClockState::Pausing(_))
    }

    pub fn start(&mut self) {
        self.state = ClockState::Running(Instant::now());
    }

    pub fn stop(&mut self) {
        self.state = ClockState::Stopping;
    }

    pub fn pause(&mut self) {
        if let ClockState::Running(start) = self.state {
            self.state = ClockState::Pausing(Instant::now() - start)
        }
    }

    pub fn resume(&mut self) {
        if let ClockState::Pausing(delta) = self.state {
            self.state = ClockState::Running(Instant::now() - delta)
        }
    }

    pub fn seek(&mut self, value: Rational) {
        let delta = value.subdivide() as u64;
        let delta = Duration::from_nanos(delta);
        self.state = ClockState::Running(Instant::now() - delta);
    }

    pub fn secs(&self) -> f64 {
        match self.state {
            ClockState::Running(start) => (Instant::now() - start).as_secs_f64(),
            _ => 0f64,
        }
    }
}

pub struct ClockAudioPlayer {
    finished: bool,
    loaded: bool,
    state: (AudioPlayerState, AudioPlayerState),
    current_rational: Rational,
    last_rational: Option<Rational>,
    duration_rational: Option<Rational>,
    sample_rate: u32,
    clock: Clock,
}

impl ClockAudioPlayer {
    pub const SMOOTHNESS_SCALE_FACTOR: u32 = 16;
    pub const DEFAULT_SAMPLE_RATE: u32 = 1440;

    pub fn new(sample_rate: u32) -> Self {
        let actual_sample_rate = sample_rate * Self::SMOOTHNESS_SCALE_FACTOR;
        Self {
            finished: false,
            loaded: false,
            state: (AudioPlayerState::Stop, AudioPlayerState::Initial),
            current_rational: Rational::new(0, actual_sample_rate),
            last_rational: None,
            duration_rational: None,
            sample_rate: actual_sample_rate,
            clock: Clock::new(),
        }
    }

    pub fn set_state(&mut self, new_state: AudioPlayerState) {
        if self.state.0 != AudioPlayerState::Initial {
            self.state.1 = self.state.0;
            self.state.0 = new_state;
        }
    }
}

impl Default for ClockAudioPlayer {
    fn default() -> Self {
        Self::new(Self::DEFAULT_SAMPLE_RATE)
    }
}

impl AudioPlayer for ClockAudioPlayer {
    fn load(bytes: &[u8]) -> Result<Self, MdanceioError>
    where
        Self: Sized,
    {
        Ok(Self::new(Self::DEFAULT_SAMPLE_RATE))
    }

    fn play(&mut self) {
        match self.state.0 {
            AudioPlayerState::Stop | AudioPlayerState::Suspend | AudioPlayerState::Pause => {
                self.finished = false;
                self.clock.start();
                self.set_state(AudioPlayerState::Play);
            }
            _ => {}
        }
    }

    fn play_part(&mut self, start: f64, length: f64) {}

    fn pause(&mut self) {
        if self.is_playing() {
            self.clock.pause();
            self.set_state(AudioPlayerState::Pause);
        }
    }

    fn resume(&mut self) {
        self.clock.resume();
        self.set_state(AudioPlayerState::Play);
    }

    fn suspend(&mut self) {
        self.set_state(AudioPlayerState::Suspend);
    }

    fn stop(&mut self) {
        if self.is_playing() || self.is_paused() {
            self.clock.stop();
            self.set_state(AudioPlayerState::Stop);
        }
    }

    fn update(&mut self) {
        let offset = (self.clock.secs() * (self.current_rational.denominator as f64)) as u64;
        self.last_rational = Some(self.current_rational);
        self.current_rational.numerator = offset;
    }

    fn seek(&mut self, rational: Rational) {
        if rational.numerator > 0 {
            self.clock.pause();
        } else {
            self.clock.stop();
        }
    }

    fn verify_bias(&mut self, expect: Rational, epsilon: Rational) -> Result<Rational, Rational> {
        self.update();
        if (self.clock.secs() - expect.subdivide()).abs() <= epsilon.subdivide() {
            Ok(self.current_rational)
        } else {
            let last = self.current_rational;
            self.seek(expect);
            Err(last)
        }
    }

    fn current_rational(&self) -> Rational {
        self.current_rational
    }

    fn last_rational(&self) -> Option<Rational> {
        self.last_rational
    }

    fn duration_rational(&self) -> Option<Rational> {
        self.duration_rational
    }

    fn bits_per_sample(&self) -> u32 {
        // (8 / 8) * 1 * self.sample_rate
        self.sample_rate
    }

    fn num_channels(&self) -> u32 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn is_playing(&self) -> bool {
        self.state.0 == AudioPlayerState::Play
    }

    fn is_paused(&self) -> bool {
        self.state.0 == AudioPlayerState::Pause
    }

    fn is_stopped(&self) -> bool {
        self.state.0 == AudioPlayerState::Stop
    }

    fn was_playing(&self) -> bool {
        self.state.0 != AudioPlayerState::Suspend && self.state.1 == AudioPlayerState::Play
    }
}
