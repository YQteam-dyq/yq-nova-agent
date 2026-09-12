use std::{
    collections::hash_map::RandomState,
    hash::{BuildHasher, Hasher},
    time::{Duration, SystemTime},
};

use crate::error::NovaResult;

struct MiniRng(u64);

impl MiniRng {
    fn new() -> Self {
        let s = RandomState::new();
        let mut h = s.build_hasher();
        h.write_u64(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x00de_adc0_ffee),
        );
        let seed = h.finish();
        MiniRng(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }

    fn range_inclusive(&mut self, lo: Duration, hi: Duration) -> Duration {
        assert!(lo <= hi, "range_inclusive: lo {lo:?} > hi {hi:?}");

        let lo_ns = lo.as_nanos() as u64;
        let hi_ns = hi.as_nanos() as u64;
        if lo_ns == hi_ns {
            return Duration::from_nanos(lo_ns);
        }
        let width = hi_ns - lo_ns + 1;

        let reject_threshold = u64::MAX - ((u64::MAX - width + 1) % width);
        loop {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            let v = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
            if v <= reject_threshold {
                return Duration::from_nanos(lo_ns + (v % width));
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryAction {
    Retry,
    Fail,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RetryConfig {
    pub max_attempts: u32,

    #[serde(with = "crate::config::duration_seconds")]
    pub base_sleep: Duration,

    #[serde(with = "crate::config::duration_seconds")]
    pub max_sleep: Duration,

    #[serde(with = "crate::config::duration_seconds")]
    pub per_attempt_timeout: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            base_sleep: Duration::from_millis(200),
            max_sleep: Duration::from_secs(2),
            per_attempt_timeout: Duration::from_secs(15),
        }
    }
}

pub async fn with_retry<T, E, F, Fut>(config: &RetryConfig, mut body: F) -> NovaResult<T>
where
    F: FnMut(u32) -> Fut,
    Fut: std::future::Future<Output = Result<T, (RetryAction, E)>>,
    E: Into<anyhow::Error> + std::fmt::Display,
{
    let mut rng = MiniRng::new();
    let mut prev = config.base_sleep;
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 1..=config.max_attempts {
        match body(attempt).await {
            Ok(v) => return Ok(v),
            Err((action, e)) => {
                let e = e.into();
                let is_last = attempt == config.max_attempts;
                let will_retry = !is_last && action == RetryAction::Retry;
                last_err = Some(e);
                if !will_retry {
                    break;
                }

                let cap = (prev * 3).min(config.max_sleep);
                let sleep = rng.range_inclusive(config.base_sleep, cap);
                prev = sleep;
                tokio::time::sleep(sleep).await;
            },
        }
    }

    let last = last_err.expect("with_retry: zero attempts ran");
    Err(crate::error::NovaError::embedding(
        format!("embedding request failed after {} attempts", config.max_attempts),
        last,
    ))
}

pub fn classify_http_status(status: reqwest::StatusCode) -> RetryAction {
    if status.as_u16() == 429 || status.is_server_error() {
        RetryAction::Retry
    } else {
        RetryAction::Fail
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[tokio::test]
    async fn retry_eventually_succeeds() {
        let cfg = RetryConfig {
            max_attempts: 3,
            base_sleep: Duration::from_millis(1),
            max_sleep: Duration::from_millis(2),
            per_attempt_timeout: Duration::from_secs(1),
        };
        let attempt = Cell::new(0u32);
        let res = with_retry(&cfg, |n| {
            attempt.set(n);
            async move {
                if n < 2 { Err((RetryAction::Retry, anyhow::anyhow!("try {n}"))) } else { Ok(n) }
            }
        })
        .await
        .unwrap();
        assert_eq!(res, 2);
        assert_eq!(attempt.get(), 2);
    }

    #[tokio::test]
    async fn retry_stops_on_fail_action() {
        let cfg = RetryConfig {
            max_attempts: 5,
            base_sleep: Duration::from_millis(1),
            max_sleep: Duration::from_millis(2),
            ..RetryConfig::default()
        };
        let res = with_retry(&cfg, |n| async move {
            if n == 1 { Err((RetryAction::Fail, anyhow::anyhow!("auth"))) } else { Ok(n) }
        })
        .await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn retry_exhausts_max_attempts() {
        let cfg = RetryConfig {
            max_attempts: 2,
            base_sleep: Duration::from_millis(1),
            max_sleep: Duration::from_millis(2),
            per_attempt_timeout: Duration::from_secs(1),
        };
        let count = Cell::new(0u32);
        let err = with_retry(&cfg, |n| {
            count.set(n);
            async move { Err::<u32, _>((RetryAction::Retry, anyhow::anyhow!("{n}"))) }
        })
        .await
        .unwrap_err();
        assert_eq!(count.get(), 2);
        assert_eq!(err.code(), crate::error::ErrorCode::Embedding);
    }

    #[test]
    fn status_classification() {
        assert_eq!(
            classify_http_status(reqwest::StatusCode::TOO_MANY_REQUESTS),
            RetryAction::Retry
        );
        assert_eq!(classify_http_status(reqwest::StatusCode::BAD_GATEWAY), RetryAction::Retry);
        assert_eq!(classify_http_status(reqwest::StatusCode::BAD_REQUEST), RetryAction::Fail);
        assert_eq!(classify_http_status(reqwest::StatusCode::UNAUTHORIZED), RetryAction::Fail);
        assert_eq!(classify_http_status(reqwest::StatusCode::OK), RetryAction::Fail);
    }
}
