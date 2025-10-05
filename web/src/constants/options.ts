export const DEPOSIT_TYPES = [
  { label: 'No Deposit', value: 'No Deposit' },
  { label: 'Non Refund', value: 'Non Refund' },
  { label: 'Refundable', value: 'Refundable' },
] as const;

export const MARKET_SEGMENTS = [
  { label: 'Direct', value: 'Direct' },
  { label: 'Corporate', value: 'Corporate' },
  { label: 'Online TA', value: 'Online TA' },
  { label: 'Offline TA/TO', value: 'Offline TA/TO' },
  { label: 'Groups', value: 'Groups' },
  { label: 'Complementary', value: 'Complementary' },
  { label: 'Aviation', value: 'Aviation' },
  { label: 'Undefined', value: 'Undefined' },
] as const;

export const EXAMPLE_BOOKING = {
  lead_time: 120,
  adr: 135.5,
  deposit_type: 'No Deposit',
  market_segment: 'Online TA',
  total_of_special_requests: 0,
  required_car_parking_spaces: 0,
  previous_cancellations: 0,
  is_repeated_guest: 0,
  adults: 2,
  stays_weekend_nights: 1,
  stays_week_nights: 3,
  arrival_month: 7,
  children: 0,
  booking_changes: 0,
};
