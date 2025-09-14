// hooks/usePlates.ts
import { useState, useEffect, useCallback } from 'react'
import { fetchPlateRecords, subscribeToPlateRecords, PlateRecord } from '@/lib/supabase.ts'

export const usePlates = () => {
  const [plates, setPlates] = useState<PlateRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Search and filter states
  const [searchTerm, setSearchTerm] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')

  // Fetch data function
  const loadPlates = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await fetchPlateRecords(searchTerm, dateFrom, dateTo)
      setPlates(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch plate records')
    } finally {
      setLoading(false)
    }
  }, [searchTerm, dateFrom, dateTo])

  // Initial data load
  useEffect(() => {
    loadPlates()
  }, [loadPlates])

  // Real-time subscription
  useEffect(() => {
    const subscription = subscribeToPlateRecords((payload) => {
      console.log('Real-time update:', payload)
      
      if (payload.eventType === 'INSERT') {
        // Transform new record to match component expectations
        const newRecord: PlateRecord = {
          ...payload.new,
          plate_text: payload.new.license_plate_number,
          captured_at: payload.new.detected_at,
          junction: payload.new.location?.junction || 'Unknown Junction',
          image_url: payload.new.crop_image_url || payload.new.full_image_url
        }
        
        // Only add if it matches current filters
        const matchesSearch = !searchTerm || 
          newRecord.license_plate_number.toLowerCase().includes(searchTerm.toLowerCase())
        
        const matchesDateFrom = !dateFrom || 
          new Date(newRecord.detected_at) >= new Date(dateFrom)
        
        const matchesDateTo = !dateTo || 
          new Date(newRecord.detected_at) <= new Date(dateTo)
        
        if (matchesSearch && matchesDateFrom && matchesDateTo) {
          setPlates(prev => [newRecord, ...prev])
        }
      } else if (payload.eventType === 'UPDATE') {
        setPlates(prev => prev.map(plate => 
          plate.id === payload.new.id 
            ? { 
                ...payload.new,
                plate_text: payload.new.license_plate_number,
                captured_at: payload.new.detected_at,
                junction: payload.new.location?.junction || 'Unknown Junction',
                image_url: payload.new.crop_image_url || payload.new.full_image_url
              }
            : plate
        ))
      } else if (payload.eventType === 'DELETE') {
        setPlates(prev => prev.filter(plate => plate.id !== payload.old.id))
      }
    })

    return () => {
      subscription.unsubscribe()
    }
  }, [searchTerm, dateFrom, dateTo])

  // Filter handlers
  const handleSearchChange = (value: string) => {
    setSearchTerm(value)
  }

  const handleDateFromChange = (value: string) => {
    setDateFrom(value)
  }

  const handleDateToChange = (value: string) => {
    setDateTo(value)
  }

  const handleClearFilters = () => {
    setSearchTerm('')
    setDateFrom('')
    setDateTo('')
  }

  // Manual refresh
  const refresh = () => {
    loadPlates()
  }

  return {
    plates,
    loading,
    error,
    searchTerm,
    dateFrom,
    dateTo,
    handleSearchChange,
    handleDateFromChange,
    handleDateToChange,
    handleClearFilters,
    refresh
  }
}