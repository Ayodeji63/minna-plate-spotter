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

// Create mock client for when Supabase is not configured
const createMockClient = () => ({
  from: () => ({
    select: () => Promise.resolve({ data: [], error: null }),
    insert: () => Promise.resolve({ data: null, error: { message: 'Supabase not configured' } }),
    order: () => ({ 
      select: () => Promise.resolve({ data: [], error: null })
    })
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