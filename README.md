# Running Locally

The frontend and backend run on your machine. The database is always the live Supabase instance — there is no local database to set up.

## Prerequisites

- [Node.js](https://nodejs.org) (v18+)
- [.NET SDK 10](https://dotnet.microsoft.com/download)

---

## 1. Configure the Backend Database Connection

Create `backend/455Chapter17.API/appsettings.Development.json` (this file is gitignored — never commit it):

```json
{
  "ConnectionStrings": {
    "Default": "Host=pooler.supabase.com;Port=5432;Database=postgres;Username=postgres.xxx;Password=yourpassword"
  }
}
```

Get your connection string from **Supabase → Settings → Database → Connection string**.

> Use the **Session mode pooler** URL (port 5432), not the direct `db.xxx.supabase.co` host.
> Format must be key-value (`Host=...;Port=...`) — not the `postgresql://` URI format.

---

## 2. Start the Backend

```bash
cd backend/455Chapter17.API
dotnet run
```

Runs at **http://localhost:5000**

Verify:
```
GET http://localhost:5000/api/fraud/ping
```

---

## 3. Start the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Runs at **http://localhost:3000**

No extra config needed — the frontend defaults to `http://localhost:5000/api` when `VITE_API_URL` is not set.

---

## Quick Reference

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:5000/api |
| Health check | http://localhost:5000/api/fraud/ping |
| Database | Supabase (always remote) |

---

## Notes

- **Vercel** hosts the production frontend. Local dev ignores it entirely.
- **Railway** hosts the production backend. Local dev runs the backend on your machine instead.
- **Supabase** is the only shared resource — both local and production point to the same database.
- If the Supabase database is empty and needs seeding, run the migration script:
  ```bash
  cd scripts
  pip install -r requirements.txt
  SUPABASE_DB_URL=<your_postgresql_uri> python import_sqlite_to_supabase.py
  ```
