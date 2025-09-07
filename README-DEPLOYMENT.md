# Vehicle Recognition System - Minna, Nigeria

A real-time license plate recognition dashboard for monitoring vehicle traffic in Minna, Niger State, Nigeria.

## 🚗 Features

- **Real-time Dashboard**: Live updates when new plates are detected
- **Smart Search**: Filter by plate number or junction name
- **Date Range Filtering**: Historical data lookup with Nigerian time formatting
- **Mobile-Friendly**: Responsive design optimized for all devices
- **Nigerian Theme**: Beautiful design inspired by Nigerian flag colors
- **Statistics Cards**: Live stats showing total plates, today's count, active junctions, and recent activity

## 🛠 Technology Stack

- **Frontend**: React + Vite + TypeScript + Tailwind CSS
- **Backend**: Supabase (Database + Real-time subscriptions)
- **UI Components**: shadcn/ui components with custom Nigerian theme
- **Real-time**: Supabase real-time subscriptions for live updates

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ and npm/pnpm
- Supabase account and project

### 1. Database Setup

1. Go to your Supabase dashboard
2. Create a new project or use existing one
3. Go to the SQL Editor and run the contents of `supabase-schema.sql`
4. This creates the `plates` table with proper RLS policies

### 2. Environment Configuration

1. Copy `.env.example` to `.env.local`
2. Fill in your Supabase credentials from your project dashboard:
   ```env
   VITE_SUPABASE_URL=https://your-project-ref.supabase.co
   VITE_SUPABASE_ANON_KEY=your-anon-key-here
   ```

### 3. Local Development

```bash
# Install dependencies
npm install
# or
pnpm install

# Start development server
npm run dev
# or
pnpm dev
```

Open [http://localhost:5173](http://localhost:5173) to view the dashboard.

## 📊 Database Schema

### `plates` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key (auto-generated) |
| `plate_text` | TEXT | License plate number (e.g., "ABC123DE") |
| `captured_at` | TIMESTAMP WITH TIME ZONE | When the plate was detected (default: now()) |
| `junction` | TEXT | Junction name (e.g., "Minna Junction A") |
| `image_url` | TEXT | Optional URL to plate image |

### Sample Data Insert (for testing)

```sql
INSERT INTO plates (plate_text, junction, image_url) VALUES 
('ABC123DE', 'Minna Junction A', 'https://example.com/plate1.jpg'),
('XYZ789FG', 'Minna Junction B', null),
('LMN456HI', 'Minna Central', 'https://example.com/plate2.jpg');
```

## 🔗 Raspberry Pi Integration

To connect your Raspberry Pi OCR system, send HTTP requests to Supabase:

```python
import requests
import json

# Your Supabase configuration
SUPABASE_URL = "https://your-project-ref.supabase.co"
SUPABASE_KEY = "your-service-role-key"  # Use service role key for Pi

def send_plate_detection(plate_text, junction, image_url=None):
    url = f"{SUPABASE_URL}/rest/v1/plates"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "plate_text": plate_text,
        "junction": junction,
        "image_url": image_url
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Example usage
send_plate_detection("ABC123DE", "Minna Junction A", "https://example.com/image.jpg")
```

## 🌐 Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Connect your GitHub repo to Vercel
3. Add environment variables in Vercel dashboard
4. Deploy automatically on every push

### Netlify

1. Build the project: `npm run build`
2. Upload the `dist` folder to Netlify
3. Configure environment variables in Netlify dashboard

### Environment Variables for Production

Set these in your deployment platform:

```env
VITE_SUPABASE_URL=https://your-project-ref.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here
```

## 🎨 Customization

### Adding New Junctions

Simply insert new records with different junction names. The system automatically detects and displays them.

### Styling

The app uses a Nigerian-themed design system defined in:
- `src/index.css` - Color variables and gradients
- `tailwind.config.ts` - Tailwind theme extension

### Real-time Features

The app automatically receives new plate detections via Supabase real-time subscriptions. No manual refresh needed!

## 📱 Mobile Experience

The dashboard is fully responsive with:
- Mobile-optimized table cards
- Touch-friendly interface
- Nigerian time formatting
- Fast loading and smooth animations

## 🔒 Security

- Row Level Security (RLS) enabled
- Public read access for dashboard
- Controlled write access for data insertion
- Environment variables for sensitive credentials

## 🆘 Troubleshooting

### No data showing?
1. Check your Supabase credentials in `.env.local`
2. Verify the `plates` table exists and has data
3. Check browser console for errors

### Real-time not working?
1. Ensure real-time is enabled in Supabase
2. Check that `ALTER PUBLICATION supabase_realtime ADD TABLE public.plates;` was run

### Build errors?
1. Ensure all dependencies are installed: `npm install`
2. Check TypeScript errors: `npm run type-check`

## 📞 Support

This system is designed for traffic monitoring in Minna, Niger State. For technical support or customizations, refer to the code documentation or create an issue in the project repository.