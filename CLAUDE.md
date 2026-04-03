# CLAUDE.md

This file documents the project structure and conventions for this repository. Use it as a template when setting up future full-stack projects.

---

## Project Overview

A full-stack web application with an integrated ML pipeline. The app has three layers:

1. **Frontend** — React SPA for end users and admins
2. **Backend** — ASP.NET Core REST API
3. **ML Pipeline** — Python/Jupyter CRISP-DM workflow

---

## Directory Structure

```
project-root/
├── frontend/                  # React + TypeScript + Vite
├── backend/
│   └── ProjectName.API/       # ASP.NET Core Web API
├── database/
│   └── supabase_schema.sql    # Production PostgreSQL schema
├── scripts/
│   ├── import_sqlite_to_supabase.py
│   └── requirements.txt
├── crispdm-pipeline-model/    # Jupyter notebooks & Python utilities
│   ├── fraud_notebook.ipynb
│   ├── pipeline.ipynb
│   └── functions.py
├── shop.db/                   # SQLite dev database
│   └── shop.db
└── ProjectName.sln            # Visual Studio solution file
```

---

## Tech Stack

### Frontend
- **React 19** with **TypeScript**
- **Vite** as build tool and dev server (port 3000)
- **React Router DOM** for SPA routing
- **ESLint + Prettier** for linting and formatting

### Backend
- **ASP.NET Core** (net10.0)
- **Entity Framework Core** with **Npgsql.EntityFrameworkCore.PostgreSQL** provider (v10.x)
- **OpenAPI/Swagger** enabled in development
- **CORS** configured to allow any origin (required for Vercel + Railway)
- Runs on **port 5000** locally; reads `PORT` env var in production (Railway sets this to 8080)
- JSON serialized as **snake_case** via `JsonNamingPolicy.SnakeCaseLower` to match frontend TypeScript interfaces

### Database
- **SQLite** (`shop.db`) — development/operational database
- **PostgreSQL via Supabase** — production schema
- **Python + psycopg** — migration tooling

### ML / Data Science
- **Python**, **Jupyter notebooks**
- CRISP-DM workflow (data understanding → deployment)
- Custom utility library in `functions.py`

---

## Frontend Structure

```
frontend/
├── src/
│   ├── pages/             # One file per route/page
│   ├── components/        # Shared UI components (Navbar, etc.)
│   ├── api/
│   │   └── client.ts      # Centralized API client + TypeScript interfaces
│   ├── App.tsx            # Router setup
│   ├── main.tsx           # React entry point
│   └── index.css          # Global styles
├── public/                # Static assets
├── vite.config.ts
├── tsconfig.json
├── .prettierrc.json
└── eslint.config.js
```

### Routing Convention
Define all routes in `App.tsx`. Each route maps 1:1 to a file in `src/pages/`.

### API Client Convention
All API calls go through `src/api/client.ts`. This file contains:
- A centralized fetch wrapper
- TypeScript interfaces for all request/response shapes
- Named functions for each endpoint

---

## Backend Structure

```
backend/
└── ProjectName.API/
    ├── Controllers/           # One controller per domain (e.g., FraudController.cs)
    ├── Program.cs             # DI setup, CORS, routing, middleware
    ├── appsettings.json       # Logging config, allowed hosts
    ├── Properties/
    │   └── launchSettings.json  # Port config, environment
    └── ProjectName.API.csproj   # NuGet dependencies
```

### CORS Setup (Program.cs)
Register a named policy (e.g., `"FrontendDev"`) that allows requests from the Vite dev server origin.

### Controller Convention
One controller file per domain area. Prefix routes with `/api/{domain}`. Always include a `/ping` health check endpoint.

---

## Database Schema

Tables in production (Supabase/PostgreSQL):

| Table | Purpose |
|-------|---------|
| `customers` | User accounts with loyalty info |
| `products` | Inventory with SKU, price, cost |
| `orders` | Order records with risk/fraud fields |
| `order_items` | Line items linking orders to products |
| `shipments` | Shipping carrier and delivery info |
| `product_reviews` | Customer ratings per product |
| `delivery_scores` | ML model predictions per order |

Add indexes on foreign keys and any column used in ORDER BY or WHERE filters.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/import_sqlite_to_supabase.py` | Migrates SQLite dev data to Supabase PostgreSQL |
| `scripts/requirements.txt` | Python deps for scripts (`psycopg[binary]`) |

### Migration Script Pattern
- Reads connection string from `SUPABASE_DB_URL` env var or `--postgres-url` CLI arg
- Normalizes types (dates, decimals, booleans) between SQLite and PostgreSQL
- Resets auto-increment sequences after import

---

## Environment Variables

No `.env` files are committed. Expected variables:

```
SUPABASE_DB_URL=postgresql://user:password@host:5432/db
ASPNETCORE_ENVIRONMENT=Production
```

`.gitignore` should exclude: `.env`, `.env.*`, `appsettings.Development.json`

---

## Package Managers

| Manager | Used For |
|---------|---------|
| `npm` | Frontend (React, TypeScript, Vite, Router) |
| `NuGet` (.csproj) | Backend (AspNetCore, EFCore, Sqlite) |
| `pip` (requirements.txt) | Python scripts |

---

## Dev Commands

### Frontend
```bash
cd frontend
npm install
npm run dev       # Start dev server at localhost:3000
npm run build     # Type check + Vite production build
npm run lint      # ESLint
npm run preview   # Preview production build locally
```

### Backend
```bash
cd backend/ProjectName.API
dotnet run        # Start API at localhost:5000
```

### Migration Script
```bash
cd scripts
pip install -r requirements.txt
SUPABASE_DB_URL=<connection_string> python import_sqlite_to_supabase.py
```

---

## Deployment

### Stack
- **Frontend** → Vercel
- **Database** → Supabase (PostgreSQL)
- **Backend** → ASP.NET Core (local or hosted separately)

---

### 1. Supabase Setup
1. Create a new project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and paste + run `database/supabase_schema.sql` to create all tables
3. Run the migration script to seed data from SQLite:
   ```bash
   cd scripts
   pip install -r requirements.txt
   SUPABASE_DB_URL=<connection_string_from_supabase_dashboard> python import_sqlite_to_supabase.py
   ```
   - Connection string is found in Supabase → **Settings → Database → Connection string (URI mode)**

---

### 2. Vercel Deployment (Frontend)
1. Push the repo to GitHub
2. Go to [vercel.com](https://vercel.com) → **Add New Project** → import the GitHub repo
3. Set the **Root Directory** to `frontend`
4. Vercel auto-detects Vite — build command is `npm run build`, output dir is `dist`
5. Add any environment variables in Vercel's **Environment Variables** settings panel
6. Deploy — Vercel provides a live URL

> If the frontend calls a hosted API, update the base URL in `src/api/client.ts` before deploying.

---

### 3. Backend (Railway)
Vercel does not support .NET — deploy the backend to **Railway** instead.

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **New Project → Deploy from GitHub repo**
3. Select your repo — if it's a fork or collaborated repo, grant Railway access via **GitHub → Settings → Applications → Authorized OAuth Apps → Railway → Grant org access**
4. Once imported, go to **Settings → Source → Root Directory** and set it to `backend/455Chapter17.API`
5. Trigger a redeploy — Railway uses Nixpacks and auto-detects .NET
6. Go to **Settings → Networking → Generate Domain** and enter port **8080** (Railway sets `PORT=8080` by default)
7. Go to **Variables** and add:
   - `ConnectionStrings__Default` = `Host=...;Port=5432;Database=postgres;Username=...;Password=...` (Npgsql format — **not** the `postgresql://` URI format, which Npgsql does not support)
8. Railway will redeploy automatically after adding variables

**Verify** by hitting `https://your-app.up.railway.app/api/fraud/ping` — should return JSON.

#### Key gotchas
- Use the **connection pooler** URL from Supabase (Session mode, port 5432) — the direct `db.xxx.supabase.co` host fails DNS resolution on Windows and Railway
- The Npgsql connection string must be key-value format: `Host=...;Port=...;Database=...;Username=...;Password=...`
- Railway sets `PORT` env var automatically — `Program.cs` must read it: `var port = Environment.GetEnvironmentVariable("PORT") ?? "5000";`
- The `.csproj` requires `<InterceptorsNamespaces>$(InterceptorsNamespaces);Microsoft.AspNetCore.OpenApi.Generated</InterceptorsNamespaces>` or the Railway build fails with a CS9137 error
- All EF Core packages and `Npgsql.EntityFrameworkCore.PostgreSQL` must be on the same major version (e.g., all 10.x)

---

### Environment Variables

Never commit secrets. Set these in each platform's dashboard:

**Railway (backend):**
```
ConnectionStrings__Default=Host=...pooler.supabase.com;Port=5432;Database=postgres;Username=postgres.xxx;Password=...
```

**Vercel (frontend):**
```
VITE_API_URL=https://your-app.up.railway.app/api
```

**Local development** — `.env` in project root:
```
SUPABASE_DB_URL=postgresql://postgres.xxx:password@pooler.supabase.com:5432/postgres
```

The frontend reads `VITE_API_URL` via `import.meta.env.VITE_API_URL` with a fallback to `http://localhost:5000/api`.

`.gitignore` should exclude: `.env`, `.env.*`, `appsettings.Development.json`

---

## Code Style

- **Prettier** config: 80 char line width, semicolons, trailing commas, single quotes
- **ESLint**: flat config, TypeScript-aware rules
- **C#**: standard ASP.NET Core conventions (PascalCase controllers, camelCase JSON)

---

## Ports (Development)

| Service | Port |
|---------|------|
| Vite dev server (frontend) | 3000 |
| ASP.NET Core API | 5000 |
| ASP.NET Core HTTPS | 7087 |
