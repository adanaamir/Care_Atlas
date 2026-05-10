import { Platform } from 'react-native';

// Public tunnel to bypass university network isolation
const LOCAL_BACKEND = 'https://hungry-cups-run.loca.lt';

// You can replace this with your HuggingFace Space URL later
export const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || LOCAL_BACKEND;


export type NeedType = 'emergency' | 'icu' | 'maternity' | 'surgery' | 'dialysis' | 'pediatric' | 'general';

export interface NearestFacilityResult {
  rank: number;
  facility_id: string;
  facility_name: string;
  composite_score: number;
  distance_km: number;
  estimated_travel_min: number;
  trust_score: number;
  trust_grade: string;
  cap_score: number;
  matched_capabilities: string[];
  facility_type: string;
  category: string;
  functional_status: string;
  facility_meta: any;
}

export interface NearestResponse {
  need_type: string;
  user_location: { lat: number; lon: number };
  radius_km: number;
  total_found: number;
  results: NearestFacilityResult[];
  fallback_used: boolean;
  message: string;
}

export async function findNearestFacilities(
  lat: number,
  lon: number,
  needType: NeedType = 'emergency',
  topK: number = 5
): Promise<NearestResponse> {
  const response = await fetch(`${API_BASE_URL}/api/nearest`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'bypass-tunnel-reminder': 'true',
    },

    body: JSON.stringify({
      lat,
      lon,
      need_type: needType,
      top_k: topK,
    }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}
