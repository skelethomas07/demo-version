# Stage 2 setup: external dataset collection with Supabase

## Files to replace/add in GitHub

Replace:
- `app.py`
- `requirements.txt`

Add:
- `supabase_schema.sql` (optional, for reference)
- `secrets.example.toml` (optional, do not rename to `secrets.toml` in GitHub)

## Supabase setup

1. Create a Supabase project.
2. Go to SQL Editor.
3. Paste and run `supabase_schema.sql`.
4. Go to Project Settings → API.
5. Copy Project URL and anon public key.

## Streamlit secrets

In Streamlit Cloud:

Manage app → Settings → Secrets

Paste:

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
anon_key = "YOUR_ANON_PUBLIC_KEY"
table = "experiment_submissions"
```

Then reboot the app.

## What this implements

- Prediction tab remains the same.
- New `Add experimental result` tab collects:
  - the same model input conditions
  - measured `Tau_ms`
  - optional source/lab/contact notes
- Rows are stored in Supabase as `pending`.
- The model is not retrained automatically yet.
