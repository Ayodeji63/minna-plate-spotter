-- License Plate Recognition Database Schema for Minna, Nigeria
-- This script sets up the plates table and necessary policies

-- Create the plates table
CREATE TABLE IF NOT EXISTS public.plates (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    plate_text TEXT NOT NULL,
    captured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    junction TEXT NOT NULL,
    image_url TEXT
);

-- Enable Row Level Security
ALTER TABLE public.plates ENABLE ROW LEVEL SECURITY;

-- Create policy for public read access
-- This allows the web app to read all plate records
CREATE POLICY "Allow public read access" ON public.plates
    FOR SELECT
    USING (true);

-- Optional: Create policy for inserting new records (for the Raspberry Pi system)
-- You may want to restrict this to authenticated users or use an API key
CREATE POLICY "Allow insert for authenticated users" ON public.plates
    FOR INSERT
    WITH CHECK (true);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS plates_captured_at_idx ON public.plates(captured_at DESC);
CREATE INDEX IF NOT EXISTS plates_junction_idx ON public.plates(junction);
CREATE INDEX IF NOT EXISTS plates_text_idx ON public.plates(plate_text);

-- Enable realtime for live updates
ALTER PUBLICATION supabase_realtime ADD TABLE public.plates;