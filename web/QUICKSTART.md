# Web Demo Quick Start

## Development

```bash
cd web
npm install
npm run dev
# Access at http://localhost:5173
```

## Production Build

```bash
npm run build
npm run preview
# Preview at http://localhost:4173
```

## Environment Variables

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` to configure:
- `VITE_API_BASE_URL` - Backend API endpoint
- `VITE_DECISION_THRESHOLD` - Low/Review threshold (default: 0.35)
- `VITE_HIGH_RISK_THRESHOLD` - Review/High threshold (default: 0.55)

## Testing with Backend API

1. Start the backend API:
```bash
cd ..
python main.py
```

2. In another terminal, start the web app:
```bash
cd web
npm run dev
```

3. Open http://localhost:5173 in your browser

## Features

- ✅ Real-time predictions
- ✅ Risk classification (High/Review/Low)
- ✅ Top 3 feature contributions
- ✅ Prediction history (last 25)
- ✅ Load example data
- ✅ Mobile-friendly layout
- ✅ Environment-based config

## Troubleshooting

**CORS errors**: Ensure the backend API has CORS middleware enabled.

**Module not found**: Run `npm install` to install dependencies.

**Build fails**: Check Node.js version (requires 18+).
