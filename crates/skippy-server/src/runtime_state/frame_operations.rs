use super::*;

fn final_sampled_chunk_start(token_count: usize, batch_size: usize) -> usize {
    debug_assert!(token_count > 0);
    (token_count - 1) / batch_size.max(1) * batch_size.max(1)
}

impl RuntimeState {
    pub fn prefill(&mut self, session_id: &str, token_ids: &[i32]) -> Result<()> {
        let session = self.session(session_id)?;
        session.prefill_chunked(token_ids)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(())
    }

    pub fn prefill_chunked_sampled(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
    ) -> Result<i32> {
        if token_ids.is_empty() {
            bail!("sampled prefill requires at least one token");
        }
        let session = self.session(session_id)?;
        let batch_size = session.batch_size()?.max(1);
        let final_chunk_start = final_sampled_chunk_start(token_ids.len(), batch_size);
        session.prefill_chunked(&token_ids[..final_chunk_start])?;
        let (predicted, _) = session.prefill_chunk_frame_sampled(
            &token_ids[final_chunk_start..],
            sampling,
            None,
            0,
        )?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(predicted)
    }

    pub fn media_marker(&self) -> String {
        self.model.media_marker()
    }

    pub fn has_media_projector(&self) -> bool {
        self.model.has_media_projector()
    }

    pub fn prefill_media(
        &mut self,
        session_id: &str,
        prompt: &str,
        media: &[MediaInput],
        sampling: Option<&SamplingConfig>,
    ) -> Result<MediaPrefill> {
        let model = &self.model as *const StageModel;
        let session = self.session(session_id)?;
        // `session()` mutably borrows the session map, while the projector lives
        // on the same RuntimeState. RuntimeState serializes access behind one
        // outer mutex, so this split borrow only aliases immutable model state.
        let prefill = unsafe { (&*model).prefill_media(session, prompt, media, sampling) }?;
        self.session_token_counts
            .insert(session_id.to_string(), prefill.position);
        Ok(prefill)
    }

    pub fn prefill_media_frame(
        &mut self,
        session_id: &str,
        prompt: &str,
        media: &[MediaInput],
    ) -> Result<MediaPrefillFrame> {
        let model = &self.model as *const StageModel;
        let session = self.session(session_id)?;
        // `session()` mutably borrows the session map, while the projector lives
        // on the same RuntimeState. RuntimeState serializes access behind one
        // outer mutex, so this split borrow only aliases immutable model state.
        let prefill = unsafe { (&*model).prefill_media_frame(session, prompt, media) }?;
        self.session_token_counts
            .insert(session_id.to_string(), prefill.position);
        Ok(prefill)
    }

    pub fn decode(&mut self, session_id: &str, token_id: i32) -> Result<i32> {
        self.decode_sampled(session_id, token_id, None)
    }

    pub fn decode_sampled(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
    ) -> Result<i32> {
        let session = self.session(session_id)?;
        let token = session.decode_step_sampled(token_id, sampling)?;
        self.add_session_tokens(session_id, 1);
        Ok(token)
    }

    pub fn decode_batch_sampled(
        &mut self,
        requests: &[RuntimeDecodeBatchRequest<'_>],
    ) -> Result<Vec<i32>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        Self::ensure_unique_batch_sessions(requests)?;
        for request in requests {
            self.session(request.session_id)?;
        }

        let mut lane_sessions = Vec::with_capacity(requests.len());
        for request in requests {
            let lane_session = self.sessions.remove(request.session_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "session {} was not active after admission",
                    request.session_id
                )
            })?;
            lane_sessions.push((request.session_id.to_string(), lane_session));
        }

        let result = {
            let mut decode_requests = lane_sessions
                .iter_mut()
                .zip(requests.iter())
                .map(|((_, lane_session), request)| DecodeBatchRequest {
                    session: &mut lane_session.session,
                    token_id: request.token_id,
                    sampling: request.sampling,
                })
                .collect::<Vec<_>>();
            StageSession::decode_batch_sampled(&mut decode_requests)
        };

        for (session_id, lane_session) in lane_sessions {
            self.sessions.insert(session_id, lane_session);
        }
        if result.is_ok() {
            for request in requests {
                self.add_session_tokens(request.session_id, 1);
            }
        }
        result
    }

    /// Returns a session's batch size, admitting a new session when necessary.
    pub fn admit_session_batch_size(&mut self, session_id: &str) -> Result<usize> {
        self.session(session_id)?.batch_size()
    }

    /// Returns the batch size of an already admitted session.
    pub fn active_session_batch_size(&mut self, session_id: &str) -> Result<usize> {
        self.active_session(session_id)?.batch_size()
    }

    pub fn ensure_session_active(&mut self, session_id: &str) -> Result<()> {
        self.session(session_id).map(|_| ())
    }

    pub fn configure_chat_sampling(
        &mut self,
        session_id: &str,
        metadata_json: &str,
        prompt_token_count: u64,
        sampling: Option<&SamplingConfig>,
    ) -> Result<()> {
        self.session(session_id)?.configure_chat_sampling(
            metadata_json,
            prompt_token_count,
            sampling,
        )
    }

    pub fn last_token_signal(&mut self, session_id: &str) -> Result<TokenSignal> {
        self.session(session_id)?.last_token_signal()
    }

    pub fn signal_window(
        &mut self,
        session_id: &str,
        window_tokens: u32,
    ) -> Result<GenerationSignalWindow> {
        self.session(session_id)?.signal_window(window_tokens)
    }

    pub fn prefill_frame(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
    ) -> Result<ActivationFrame> {
        self.prefill_frame_with_positions(session_id, token_ids, &[], input)
    }

    pub fn prefill_frame_with_positions(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        positions: &[i32],
        input: Option<&ActivationFrame>,
    ) -> Result<ActivationFrame> {
        let session = self.session(session_id)?;
        let frame = session.prefill_chunk_frame_with_positions(token_ids, positions, input, 0)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(frame)
    }

    pub fn prefill_final_frame_sampled(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        positions: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
    ) -> Result<(i32, ActivationFrame)> {
        let session = self.session(session_id)?;
        let (predicted, frame) = session
            .prefill_chunk_frame_sampled_with_positions(token_ids, positions, sampling, input, 0)?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok((predicted, frame))
    }

    pub fn decode_frame(
        &mut self,
        session_id: &str,
        token_id: i32,
        input: Option<&ActivationFrame>,
    ) -> Result<(i32, ActivationFrame)> {
        self.decode_frame_sampled(session_id, token_id, None, input, 0)
    }

    pub fn decode_frame_sampled(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(i32, ActivationFrame)> {
        let session = self.session(session_id)?;
        let output =
            session.decode_step_frame_sampled(token_id, sampling, input, output_capacity)?;
        self.add_session_tokens(session_id, 1);
        Ok(output)
    }

    pub fn decode_frame_sampled_mtp(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
        max_draft_tokens: usize,
    ) -> Result<(i32, Option<NativeMtpDraft>, ActivationFrame)> {
        let session = self.session(session_id)?;
        let output = session.decode_step_frame_sampled_mtp(
            token_id,
            sampling,
            input,
            output_capacity,
            max_draft_tokens,
        )?;
        self.add_session_tokens(session_id, 1);
        Ok(output)
    }

    pub fn decode_sampled_mtp(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        max_draft_tokens: usize,
    ) -> Result<(i32, Option<NativeMtpDraft>)> {
        let session = self.session(session_id)?;
        let output = session.decode_step_sampled_mtp(token_id, sampling, max_draft_tokens)?;
        self.add_session_tokens(session_id, 1);
        Ok(output)
    }

    pub fn decode_frame_batch_sampled(
        &mut self,
        requests: &[RuntimeDecodeFrameBatchRequest<'_>],
    ) -> Result<Vec<DecodeFrameBatchOutput>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        Self::ensure_unique_frame_batch_sessions(requests)?;
        for request in requests {
            self.session(request.session_id)?;
        }

        let mut lane_sessions = Vec::with_capacity(requests.len());
        for request in requests {
            let lane_session = self.sessions.remove(request.session_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "session {} was not active after admission",
                    request.session_id
                )
            })?;
            lane_sessions.push((request.session_id.to_string(), lane_session));
        }

        let result = {
            let mut decode_requests = lane_sessions
                .iter_mut()
                .zip(requests.iter())
                .map(|((_, lane_session), request)| DecodeFrameBatchRequest {
                    session: &mut lane_session.session,
                    token_id: request.token_id,
                    sampling: request.sampling,
                    input: request.input,
                })
                .collect::<Vec<_>>();
            StageSession::decode_step_frame_batch_sampled(&mut decode_requests)
        };

        for (session_id, lane_session) in lane_sessions {
            self.sessions.insert(session_id, lane_session);
        }
        if result.is_ok() {
            for request in requests {
                self.add_session_tokens(request.session_id, 1);
            }
        }
        result
    }

    pub fn iteration_batch_sampled(
        &mut self,
        requests: &[RuntimeIterationBatchRequest<'_>],
    ) -> Result<IterationBatchOutput> {
        if requests.is_empty() {
            return Ok(IterationBatchOutput {
                request_outputs: Vec::new(),
                samples: Vec::new(),
            });
        }
        let mut unique = std::collections::BTreeSet::new();
        for request in requests {
            if !unique.insert(request.session_id) {
                bail!(
                    "iteration contains duplicate session {}",
                    request.session_id
                );
            }
        }
        let new_session_count = unique
            .iter()
            .filter(|session_id| !self.sessions.contains_key(**session_id))
            .count();
        let available_lanes = (self.lane_count as usize).saturating_sub(self.sessions.len());
        ensure_iteration_session_capacity(new_session_count, available_lanes)?;
        for request in requests {
            self.session(request.session_id)?;
        }

        let mut lane_sessions = Vec::with_capacity(requests.len());
        for request in requests {
            let lane_session = self.sessions.remove(request.session_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "session {} was not active after admission",
                    request.session_id
                )
            })?;
            lane_sessions.push((request.session_id.to_string(), lane_session));
        }

        let result = {
            let mut iteration_requests = lane_sessions
                .iter_mut()
                .zip(requests.iter())
                .map(|((_, lane_session), request)| IterationBatchRequest {
                    session: &mut lane_session.session,
                    token_ids: request.token_ids,
                    positions: request.positions,
                    sampling: request.sampling,
                    input: request.input,
                    sample_last: request.sample_last,
                    phase: request.phase,
                })
                .collect::<Vec<_>>();
            StageSession::iteration_batch_sampled(&mut iteration_requests)
        };

        for (session_id, lane_session) in lane_sessions {
            self.sessions.insert(session_id, lane_session);
        }
        if result.is_ok() {
            for request in requests {
                self.add_session_tokens(request.session_id, request.token_ids.len() as u64);
            }
        }
        result
    }

    pub fn verify_frame(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(Vec<i32>, Option<NativeMtpDraft>, ActivationFrame)> {
        self.verify_frame_sampled(session_id, token_ids, None, input, output_capacity, 0)
    }

    pub(crate) fn canonical_session_position(&self, session_id: &str) -> Result<u64> {
        let tracked_position = self
            .session_token_counts
            .get(session_id)
            .copied()
            .with_context(|| format!("session {session_id} has no tracked position"))?;
        let session = self
            .sessions
            .get(session_id)
            .with_context(|| format!("session {session_id} is not active"))?;
        let rust_position = session.session.token_count();
        let native_position = session.session.native_position()?;
        if tracked_position != rust_position || tracked_position != native_position {
            bail!(
                "session {session_id} position mismatch: tracked={tracked_position}, rust={rust_position}, native={native_position}"
            );
        }
        Ok(native_position)
    }

    pub(crate) fn verify_tokens_sampled(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
    ) -> Result<Vec<i32>> {
        let token_count = u64::try_from(token_ids.len())
            .context("linear verification token count exceeds u64")?;
        let session = self.session(session_id)?;
        let predicted = session.verify_tokens_sampled(token_ids, sampling)?;
        self.add_session_tokens(session_id, token_count);
        Ok(predicted)
    }

    /// Verifies a speculative span in one batched forward, returning the target
    /// predictions plus the MTP draft for the branch that was verified.
    pub(crate) fn verify_tokens_sampled_mtp(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
        max_draft_tokens: usize,
    ) -> Result<(Vec<i32>, Option<NativeMtpDraft>)> {
        let token_count = u64::try_from(token_ids.len())
            .context("native MTP verification token count exceeds u64")?;
        let session = self.session(session_id)?;
        let (predicted, draft) =
            session.verify_tokens_sampled_mtp(token_ids, sampling, max_draft_tokens)?;
        self.add_session_tokens(session_id, token_count);
        Ok((predicted, draft))
    }

    pub(crate) fn session_token_count(&self, session_id: &str) -> Option<u64> {
        self.session_token_counts.get(session_id).copied()
    }

    pub fn verify_frame_sampled(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
        max_draft_tokens: usize,
    ) -> Result<(Vec<i32>, Option<NativeMtpDraft>, ActivationFrame)> {
        let session = self.session(session_id)?;
        let output = session.verify_tokens_frame_sampled(
            token_ids,
            sampling,
            input,
            output_capacity,
            max_draft_tokens,
        )?;
        self.add_session_tokens(session_id, token_ids.len() as u64);
        Ok(output)
    }

    pub fn verify_frame_sampled_serial(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
    ) -> Result<(Vec<i32>, Option<NativeMtpDraft>, ActivationFrame)> {
        if token_ids.is_empty() {
            bail!("serial verify_frame requires at least one token");
        }
        let input_frames = split_activation_frame(input, token_ids.len())?;
        let mut predicted_tokens = Vec::with_capacity(token_ids.len());
        let mut output_frames = Vec::with_capacity(token_ids.len());
        let mut last_draft = None;
        for (index, token_id) in token_ids.iter().copied().enumerate() {
            let input_frame = input_frames.as_ref().map(|frames| &frames[index]);
            let (predicted, native_mtp, output) = self.decode_frame_sampled_mtp(
                session_id,
                token_id,
                sampling,
                input_frame,
                output_capacity,
                1,
            )?;
            if predicted >= 0 {
                predicted_tokens.push(predicted);
            }
            last_draft = native_mtp;
            output_frames.push(output);
        }
        Ok((
            predicted_tokens,
            last_draft,
            combine_activation_frames(&output_frames)?,
        ))
    }

    pub fn retire_verify_checkpoint(
        &mut self,
        session_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> Result<()> {
        self.active_session(session_id)?
            .retire_verify_checkpoint(token_start, token_count)
    }

    pub fn trim_session(&mut self, session_id: &str, token_count: u64) -> Result<()> {
        let session = self.session(session_id)?;
        session.trim_session(token_count)?;
        self.session_token_counts
            .insert(session_id.to_string(), token_count);
        self.notify_session_lifecycle(super::lifecycle::SessionLifecycleEvent::SessionTrimmed {
            token_count,
        });
        Ok(())
    }

    pub fn align_session_to_token_count_if_ahead(
        &mut self,
        session_id: &str,
        token_count: u64,
    ) -> Result<Option<RuntimeSessionAlignStats>> {
        let Some(current) = self.session_token_counts.get(session_id).copied() else {
            return Ok(None);
        };
        if current <= token_count {
            return Ok(None);
        }
        self.trim_session(session_id, token_count)?;
        Ok(Some(RuntimeSessionAlignStats {
            before_token_count: current,
            after_token_count: token_count,
        }))
    }

    pub(super) fn session(&mut self, session_id: &str) -> Result<&mut StageSession> {
        if !self.sessions.contains_key(session_id) {
            let lane_session = self.take_idle_session().map(Ok).unwrap_or_else(|| {
                if self.sessions.len() >= self.lane_count as usize {
                    bail!("all execution lanes are busy");
                }
                self.create_lane_session()
            })?;
            self.sessions.insert(session_id.to_string(), lane_session);
            // Every active session must have a tracked position from the
            // moment it activates (both a fresh and a reused idle lane start
            // at native position 0), so `canonical_session_position` is
            // defined even for a generation that completes before its first
            // prefill/decode call -- see the KV-disabled repro this fixes
            // (`session ... has no tracked position`).
            self.session_token_counts
                .entry(session_id.to_string())
                .or_insert(0);
        }
        Ok(&mut self
            .sessions
            .get_mut(session_id)
            .expect("session inserted above")
            .session)
    }

    fn ensure_unique_batch_sessions(requests: &[RuntimeDecodeBatchRequest<'_>]) -> Result<()> {
        let mut seen = BTreeSet::new();
        for request in requests {
            if !seen.insert(request.session_id) {
                bail!("duplicate session {} in decode batch", request.session_id);
            }
        }
        Ok(())
    }

    fn ensure_unique_frame_batch_sessions(
        requests: &[RuntimeDecodeFrameBatchRequest<'_>],
    ) -> Result<()> {
        let mut seen = BTreeSet::new();
        for request in requests {
            if !seen.insert(request.session_id) {
                bail!(
                    "duplicate session {} in decode frame batch",
                    request.session_id
                );
            }
        }
        Ok(())
    }

    pub(super) fn active_session(&mut self, session_id: &str) -> Result<&mut StageSession> {
        self.sessions
            .get_mut(session_id)
            .map(|lane_session| &mut lane_session.session)
            .ok_or_else(|| anyhow::anyhow!("session {session_id} is not active"))
    }
}

#[cfg(test)]
mod tests {
    use super::final_sampled_chunk_start;

    #[test]
    fn sampled_prefill_keeps_the_final_native_chunk_intact() {
        assert_eq!(final_sampled_chunk_start(1, 512), 0);
        assert_eq!(final_sampled_chunk_start(511, 512), 0);
        assert_eq!(final_sampled_chunk_start(512, 512), 0);
        assert_eq!(final_sampled_chunk_start(513, 512), 512);
        assert_eq!(final_sampled_chunk_start(6603, 2048), 6144);
    }
}

fn ensure_iteration_session_capacity(
    new_session_count: usize,
    available_lanes: usize,
) -> Result<()> {
    if new_session_count > available_lanes {
        bail!(
            "iteration requires {new_session_count} new sessions but only {available_lanes} execution lanes are available"
        );
    }
    Ok(())
}

fn split_activation_frame(
    input: Option<&ActivationFrame>,
    token_count: usize,
) -> Result<Option<Vec<ActivationFrame>>> {
    let Some(input) = input else {
        return Ok(None);
    };
    if token_count == 0 {
        bail!("cannot split activation frame for zero tokens");
    }
    if input.desc.token_count as usize != token_count {
        bail!(
            "activation token count mismatch: frame={} tokens={}",
            input.desc.token_count,
            token_count
        );
    }
    if input.payload.len() % token_count != 0 {
        bail!(
            "activation payload is not divisible by token count: payload={} tokens={}",
            input.payload.len(),
            token_count
        );
    }
    let row_bytes = input.payload.len() / token_count;
    let frames = input
        .payload
        .chunks(row_bytes)
        .map(|row| {
            let mut desc = input.desc;
            desc.token_count = 1;
            desc.sequence_count = 1;
            desc.payload_bytes = row.len() as u64;
            ActivationFrame {
                desc,
                payload: row.to_vec(),
            }
        })
        .collect();
    Ok(Some(frames))
}

fn combine_activation_frames(frames: &[ActivationFrame]) -> Result<ActivationFrame> {
    let Some(first) = frames.first() else {
        bail!("cannot combine empty activation frames");
    };
    let mut desc = first.desc;
    let mut payload = Vec::new();
    let mut token_count = 0u32;
    for frame in frames {
        if frame.desc.dtype != desc.dtype
            || frame.desc.layout != desc.layout
            || frame.desc.producer_stage_index != desc.producer_stage_index
            || frame.desc.layer_start != desc.layer_start
            || frame.desc.layer_end != desc.layer_end
            || frame.desc.sequence_count != desc.sequence_count
            || frame.desc.flags != desc.flags
        {
            bail!("cannot combine incompatible activation frames");
        }
        token_count = token_count
            .checked_add(frame.desc.token_count)
            .context("combined activation token count overflow")?;
        payload.extend_from_slice(&frame.payload);
    }
    desc.token_count = token_count;
    desc.payload_bytes = payload.len() as u64;
    Ok(ActivationFrame { desc, payload })
}

#[cfg(test)]
mod iteration_admission_tests {
    use super::*;

    #[test]
    fn rejects_batch_before_partial_session_admission() {
        assert!(ensure_iteration_session_capacity(3, 2).is_err());
        assert!(ensure_iteration_session_capacity(2, 2).is_ok());
    }
}
