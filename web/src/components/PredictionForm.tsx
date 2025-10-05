import { useState } from 'react';
import { DEPOSIT_TYPES, MARKET_SEGMENTS, EXAMPLE_BOOKING } from '../constants/options';
import type { BookingInput } from '../types/api';

interface PredictionFormProps {
  onSubmit: (booking: BookingInput) => void;
  loading: boolean;
}

export function PredictionForm({ onSubmit, loading }: PredictionFormProps) {
  const [formData, setFormData] = useState<BookingInput>({
    lead_time: 0,
    arrival_month: 1,
    stays_weekend_nights: 0,
    stays_week_nights: 0,
    adults: 1,
    children: 0,
    is_repeated_guest: 0,
    previous_cancellations: 0,
    booking_changes: 0,
    adr: 0,
    required_car_parking_spaces: 0,
    total_of_special_requests: 0,
  });

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) || 0 : value,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const handleLoadExample = () => {
    setFormData(EXAMPLE_BOOKING);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Lead Time */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Lead Time (days)
          </label>
          <input
            type="number"
            name="lead_time"
            value={formData.lead_time}
            onChange={handleChange}
            min="0"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* ADR */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Average Daily Rate (ADR)
          </label>
          <input
            type="number"
            name="adr"
            value={formData.adr}
            onChange={handleChange}
            min="0"
            step="0.01"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Deposit Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Deposit Type
          </label>
          <select
            name="deposit_type"
            value={formData.deposit_type || ''}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Select...</option>
            {DEPOSIT_TYPES.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Market Segment */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Market Segment
          </label>
          <select
            name="market_segment"
            value={formData.market_segment || ''}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Select...</option>
            {MARKET_SEGMENTS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Special Requests */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Special Requests
          </label>
          <input
            type="number"
            name="total_of_special_requests"
            value={formData.total_of_special_requests}
            onChange={handleChange}
            min="0"
            max="5"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Parking Spaces */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Parking Spaces
          </label>
          <input
            type="number"
            name="required_car_parking_spaces"
            value={formData.required_car_parking_spaces}
            onChange={handleChange}
            min="0"
            max="3"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Previous Cancellations */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Previous Cancellations
          </label>
          <input
            type="number"
            name="previous_cancellations"
            value={formData.previous_cancellations}
            onChange={handleChange}
            min="0"
            max="5"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Repeated Guest */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Repeated Guest
          </label>
          <select
            name="is_repeated_guest"
            value={formData.is_repeated_guest}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={0}>No</option>
            <option value={1}>Yes</option>
          </select>
        </div>

        {/* Adults */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Adults
          </label>
          <input
            type="number"
            name="adults"
            value={formData.adults}
            onChange={handleChange}
            min="1"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Children */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Children
          </label>
          <input
            type="number"
            name="children"
            value={formData.children}
            onChange={handleChange}
            min="0"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Weekend Nights */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Weekend Nights
          </label>
          <input
            type="number"
            name="stays_weekend_nights"
            value={formData.stays_weekend_nights}
            onChange={handleChange}
            min="0"
            max="5"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Week Nights */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Week Nights
          </label>
          <input
            type="number"
            name="stays_week_nights"
            value={formData.stays_week_nights}
            onChange={handleChange}
            min="0"
            max="10"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Arrival Month */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Arrival Month
          </label>
          <input
            type="number"
            name="arrival_month"
            value={formData.arrival_month}
            onChange={handleChange}
            min="1"
            max="12"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Booking Changes */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Booking Changes
          </label>
          <input
            type="number"
            name="booking_changes"
            value={formData.booking_changes}
            onChange={handleChange}
            min="0"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      <div className="flex gap-3 pt-4">
        <button
          type="submit"
          disabled={loading}
          className="flex-1 bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
        >
          {loading ? 'Predicting...' : 'Predict Cancellation'}
        </button>
        <button
          type="button"
          onClick={handleLoadExample}
          className="px-6 py-2 border border-gray-300 rounded-md hover:bg-gray-50 font-medium"
        >
          Load Example
        </button>
      </div>
    </form>
  );
}
