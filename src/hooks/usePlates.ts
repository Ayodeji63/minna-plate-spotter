import { useState, useEffect } from 'react';
import { supabase, type PlateRecord } from '@/lib/supabase';
import { useToast } from '@/hooks/use-toast';

export const usePlates = () => {
  const [plates, setPlates] = useState<PlateRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  // Fetch plates from database
  const fetchPlates = async () => {
    try {
      const { data, error } = await supabase
        .from('plates')
        .select('*')
        .order('captured_at', { ascending: false });

      if (error) throw error;
      
      setPlates(data || []);
    } catch (error) {
      console.error('Error fetching plates:', error);
      toast({
        title: 'Error',
        description: 'Failed to fetch plate records. Please check your connection.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  // Set up real-time subscription
  useEffect(() => {
    fetchPlates();

    // Subscribe to real-time changes
    const subscription = supabase
      .channel('plates-changes')
      .on(
        'postgres_changes',
        {
          event: '*', // Listen to all changes (INSERT, UPDATE, DELETE)
          schema: 'public',
          table: 'plates',
        },
        (payload) => {
          console.log('Real-time update:', payload);
          
          if (payload.eventType === 'INSERT') {
            const newPlate = payload.new as PlateRecord;
            setPlates((prev) => [newPlate, ...prev]);
            
            toast({
              title: 'New Plate Detected!',
              description: `Plate ${newPlate.plate_text} detected at ${newPlate.junction}`,
            });
          } else if (payload.eventType === 'UPDATE') {
            const updatedPlate = payload.new as PlateRecord;
            setPlates((prev) =>
              prev.map((plate) =>
                plate.id === updatedPlate.id ? updatedPlate : plate
              )
            );
          } else if (payload.eventType === 'DELETE') {
            const deletedPlate = payload.old as PlateRecord;
            setPlates((prev) =>
              prev.filter((plate) => plate.id !== deletedPlate.id)
            );
          }
        }
      )
      .subscribe();

    // Cleanup subscription on unmount
    return () => {
      subscription.unsubscribe();
    };
  }, [toast]);

  return { plates, loading, refetch: fetchPlates };
};