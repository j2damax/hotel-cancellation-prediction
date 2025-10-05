/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'risk-high': '#dc2626',
        'risk-review': '#f59e0b',
        'risk-low': '#059669',
      },
    },
  },
  plugins: [],
}
