-- Synapse Retention Engine: pending experiment submissions table
-- Run this in Supabase SQL Editor.

create extension if not exists pgcrypto;

create table if not exists public.experiment_submissions (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),

    status text not null default 'pending' check (status in ('pending', 'approved', 'rejected')),
    source text not null default 'user_submission',

    submitter_name text,
    submitter_email text,
    organization text,
    source_reference text,
    notes text,

    -- Required measured label for future model training
    tau_ms double precision not null check (tau_ms > 0),

    -- Current model prediction at the time of submission, for later error analysis
    pred_tau_ms double precision,
    pred_log1p_tau_ms double precision,

    -- Full model input row as JSON. Approved rows can later be converted into the training table schema.
    input_data jsonb not null,
    input_completeness_filled integer,
    input_completeness_total integer,
    model_type text,

    review_comment text,
    reviewed_at timestamptz
);

alter table public.experiment_submissions enable row level security;

-- Allow the Streamlit app, using the anon public key, to insert pending rows only.
drop policy if exists "allow_anon_insert_pending_submissions" on public.experiment_submissions;
create policy "allow_anon_insert_pending_submissions"
on public.experiment_submissions
for insert
to anon
with check (status = 'pending');

-- Do not add public select/update/delete policies for the beta.
-- Review submissions inside the Supabase dashboard or with a private admin script.
