import { createClient } from '@supabase/supabase-js';

// Try different possible environment variable names for Lovable's Supabase integration
const possibleUrls = [
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.SUPABASE_URL,
  import.meta.env.VITE_PUBLIC_SUPABASE_URL,
  import.meta.env.PUBLIC_SUPABASE_URL
];

const possibleKeys = [
  import.meta.env.VITE_SUPABASE_ANON_KEY,
  import.meta.env.SUPABASE_ANON_KEY,
  import.meta.env.VITE_PUBLIC_SUPABASE_ANON_KEY,
  import.meta.env.PUBLIC_SUPABASE_ANON_KEY
];

const supabaseUrl = possibleUrls.find(url => url);
const supabaseAnonKey = possibleKeys.find(key => key);

console.log('Trying to connect to Supabase...');
console.log('Available env vars:', Object.keys(import.meta.env));
console.log('Found URL:', supabaseUrl ? 'Yes' : 'No');
console.log('Found Key:', supabaseAnonKey ? 'Yes' : 'No');

// Create mock data for demonstration
const createMockData = () => {
  const currentTime = new Date();
  const junctions = [
    'Minna Central Junction',
    'Minna Junction A',
    'Minna Junction B',
    'Bosso Junction',
    'Chanchaga Junction',
    'Tunga Junction',
    'Dutsen Kura Junction',
    'Kpakungu Junction'
  ];
  
  const plateFormats = ['ABC', 'XYZ', 'LMN', 'DEF', 'GHI', 'JKL', 'RST', 'UVW'];
  const plateNumbers = ['123', '456', '789', '012', '345', '678', '901', '234'];
  const plateSuffixes = ['DE', 'FG', 'HI', 'JK', 'LM', 'NO', 'PQ', 'RS'];

  return Array.from({ length: 15 }, (_, i) => {
    const minutesAgo = Math.floor(Math.random() * 1440); // Random time within last 24 hours
    const capturedTime = new Date(currentTime.getTime() - minutesAgo * 60000);
    
    return {
      id: `mock-${i + 1}`,
      plate_text: `${plateFormats[i % plateFormats.length]}${plateNumbers[i % plateNumbers.length]}${plateSuffixes[i % plateSuffixes.length]}`,
      captured_at: capturedTime.toISOString(),
      junction: junctions[i % junctions.length],
      image_url: Math.random() > 0.3 ? `https://via.placeholder.com/120x80/2D7D32/FFFFFF?text=Plate+${i + 1}` : null
    };
  }).sort((a, b) => new Date(b.captured_at).getTime() - new Date(a.captured_at).getTime());
};

const mockData = createMockData();

// Create mock client for when Supabase is not configured
const createMockClient = () => ({
  from: () => ({
    select: () => ({
      order: () => Promise.resolve({ data: mockData, error: null })
    }),
    insert: () => Promise.resolve({ data: null, error: { message: 'Supabase not configured - this is demo data' } }),
  }),
  channel: () => ({
    on: () => ({ subscribe: () => ({ unsubscribe: () => {} }) })
  })
});

// Export the Supabase client
export const supabase = supabaseUrl && supabaseAnonKey 
  ? createClient(supabaseUrl, supabaseAnonKey)
  : createMockClient() as any;

// Log configuration status
if (!supabaseUrl || !supabaseAnonKey) {
  console.error('Supabase not configured yet. Using mock client. Please set up your environment variables.');
} else {
  console.log('Supabase client configured successfully!');
}

export type Database = {
  public: {
    Tables: {
      plates: {
        Row: {
          id: string;
          plate_text: string;
          captured_at: string;
          junction: string;
          image_url: string | null;
        };
        Insert: {
          id?: string;
          plate_text: string;
          captured_at?: string;
          junction: string;
          image_url?: string | null;
        };
        Update: {
          id?: string;
          plate_text?: string;
          captured_at?: string;
          junction?: string;
          image_url?: string | null;
        };
      };
    };
  };
};

export type PlateRecord = Database['public']['Tables']['plates']['Row'];