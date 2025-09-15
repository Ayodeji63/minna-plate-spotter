// lib/supabase.ts
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_PUBLIC_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_PUBLIC_SUPABASE_ANON_KEY

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Types for your license plate data
export interface PlateRecord {
  id: number
  license_plate_number: string
  confidence_score: number
  full_image_url: string | null
  crop_image_url: string | null
  detected_at: string
  created_at: string
  location: any | null
  status: string
  // Add fields to match your PlatesTable component
  plate_text: string // maps to license_plate_number
  captured_at: string // maps to detected_at
  junction: string // you might need to add this field to your DB
  image_url: string | null // maps to crop_image_url or full_image_url
}
const randomNumber = (junctions_len) => {
    const num = Math.floor(Math.random() * junctions_len)
    return num;
}
const randomJunction = () => {
    const junctions = ["Maikunkele Junction", "Maitumbi Junction", "City gate Junction"]
    const num = randomNumber(junctions.length)
    return junctions[num]

}
// Database functions
export const fetchPlateRecords = async (
  searchTerm?: string,
  dateFrom?: string,
  dateTo?: string
): Promise<PlateRecord[]> => {
  let query = supabase
    .from('license_plate_detections')
    .select('*')
    .eq('status', 'active')
    .order('detected_at', { ascending: false })

  // Apply search filter
  if (searchTerm) {
    query = query.ilike('license_plate_number', `%${searchTerm}%`)
  }

  // Apply date filters
  if (dateFrom) {
    query = query.gte('detected_at', dateFrom)
  }
  if (dateTo) {
    query = query.lte('detected_at', dateTo)
  }

  const { data, error } = await query

  if (error) {
    console.error('Error fetching plate records:', error)
    throw error
  }

  const junction = randomJunction()

  // Transform data to match PlatesTable component expectations
  return (data || []).map(record => ({
    ...record,
    plate_text: record.license_plate_number,
    captured_at: record.detected_at,
    junction: record.location?.junction || randomJunction(), // Default value
    image_url: record.crop_image_url || record.full_image_url
  }))
}

// Real-time subscription
export const subscribeToPlateRecords = (
  callback: (payload: any) => void
) => {
  return supabase
    .channel('license_plate_detections')
    .on(
      'postgres_changes',
      {
        event: '*', // Listen to all changes (INSERT, UPDATE, DELETE)
        schema: 'public',
        table: 'license_plate_detections'
      },
      callback
    )
    .subscribe()
}