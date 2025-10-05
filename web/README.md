# Hotel Cancellation Prediction - Web Demo

A minimal, production-style demo UI for the Hotel Cancellation Prediction API.

## Features

- **Single-page React application** with TypeScript and Tailwind CSS
- **Real-time predictions** via FastAPI backend
- **Risk classification** (High/Review/Low) based on configurable thresholds
- **Feature importance visualization** using SHAP data from the backend
- **Prediction history** tracking (last 25 predictions)
- **Mobile-friendly layout** with responsive design
- **Environment-configurable** API base URL and decision thresholds

## Tech Stack

- React 18 + TypeScript
- Vite (build tooling)
- Tailwind CSS (styling)
- Custom hooks for state management
- Fetch API for backend communication

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Hotel Cancellation Prediction API running (see parent directory README)

### Installation

```bash
cd web
npm install
```

### Configuration

Copy the example environment file and update as needed:

```bash
cp .env.example .env
```

Environment variables:

- `VITE_API_BASE_URL` - Backend API URL (default: `http://localhost:8000`)
- `VITE_DECISION_THRESHOLD` - Threshold for Low/Review classification (default: `0.35`)
- `VITE_HIGH_RISK_THRESHOLD` - Threshold for Review/High classification (default: `0.55`)

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:5173` by default.

### Production Build

```bash
npm run build
npm run preview
```

Build artifacts are generated in the `dist/` directory.

## Usage

1. **Start the backend API** (from parent directory):
   ```bash
   python main.py
   ```

2. **Fill in booking details** in the form or click "Load Example"

3. **Submit prediction** to see:
   - Cancellation probability
   - Risk classification (High/Review/Low)
   - Top 3 feature contributions
   - Prediction history chart

## Data Contract

The form submits to `POST /predict` with the following fields:

```typescript
{
  lead_time: number;
  arrival_month: number;
  stays_weekend_nights: number;
  stays_week_nights: number;
  adults: number;
  children: number;
  is_repeated_guest: 0 | 1;
  previous_cancellations: number;
  booking_changes: number;
  adr: number;
  required_car_parking_spaces: number;
  total_of_special_requests: number;
  deposit_type?: string;
  market_segment?: string;
}
```

## Deployment

### Static Hosting (AWS S3 + CloudFront)

```bash
# Build for production
npm run build

# Deploy to S3
aws s3 sync dist/ s3://your-bucket-name/ --delete

# Invalidate CloudFront cache (optional)
aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths '/*'
```

### Docker (via parent directory)

The web app can be served alongside the API using the parent directory's Docker setup.

## Project Structure

```
web/
├── src/
│   ├── components/          # React components
│   │   ├── PredictionForm.tsx
│   │   ├── ResultPanel.tsx
│   │   ├── RiskBadge.tsx
│   │   ├── FeatureContributions.tsx
│   │   └── ProbabilityHistoryChart.tsx
│   ├── hooks/              # Custom React hooks
│   │   ├── usePredictionHistory.ts
│   │   └── useInterpretability.ts
│   ├── lib/                # Utilities
│   │   ├── api.ts          # API client
│   │   └── risk.ts         # Risk classification
│   ├── constants/          # Configuration
│   │   ├── options.ts      # Form options
│   │   └── thresholds.ts   # Decision thresholds
│   ├── types/              # TypeScript types
│   │   └── api.ts
│   ├── App.tsx             # Main application
│   ├── main.tsx            # Entry point
│   └── index.css           # Tailwind imports
├── .env.example            # Environment template
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## Bundle Size

- Initial JS bundle: ~209 KB (gzip: 64 KB)
- CSS: ~3 KB (gzip: <1 KB)
- Total: < 300 KB (meets acceptance criteria)

## Error Handling

- Network failures show inline error messages
- Form validation prevents invalid inputs
- Graceful fallbacks when interpretability data is unavailable
- 10-second timeout on API requests

## Browser Support

- Modern browsers (Chrome, Firefox, Safari, Edge)
- ES2020+ JavaScript features required

## License

See parent directory LICENSE file.
