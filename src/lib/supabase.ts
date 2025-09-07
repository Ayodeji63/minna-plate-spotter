import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

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