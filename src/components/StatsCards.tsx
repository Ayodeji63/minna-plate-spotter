import { Car, Clock, MapPin, TrendingUp } from 'lucide-react';
import { Card } from '@/components/ui/card';
import type { PlateRecord } from '@/lib/supabase.ts';

interface StatsCardsProps {
  plates: PlateRecord[];
}

const StatsCards = ({ plates }: StatsCardsProps) => {
  const totalPlates = plates.length;
  const todayPlates = plates.filter(plate => {
    const today = new Date();
    const plateDate = new Date(plate.captured_at);
    return plateDate.toDateString() === today.toDateString();
  }).length;
  
  const uniqueJunctions = new Set(plates.map(plate => plate.junction)).size;
  
  const recentPlates = plates.filter(plate => {
    const now = new Date();
    const plateDate = new Date(plate.captured_at);
    const hourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    return plateDate >= hourAgo;
  }).length;

  const stats = [
    {
      title: 'Total Plates',
      value: totalPlates.toLocaleString(),
      icon: Car,
      color: 'text-nigeria-green',
      bg: 'bg-nigeria-green/10',
    },
    {
      title: 'Today',
      value: todayPlates.toLocaleString(),
      icon: Clock,
      color: 'text-info',
      bg: 'bg-blue-50',
    },
    {
      title: 'Active Junctions',
      value: uniqueJunctions.toString(),
      icon: MapPin,
      color: 'text-warning',
      bg: 'bg-orange-50',
    },
    {
      title: 'Last Hour',
      value: recentPlates.toString(),
      icon: TrendingUp,
      color: 'text-success',
      bg: 'bg-green-50',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {stats.map((stat) => {
        const Icon = stat.icon;
        return (
          <Card key={stat.title} className="p-6 bg-gradient-card shadow-card hover:shadow-hover transition-shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">{stat.title}</p>
                <p className="text-2xl font-bold text-foreground">{stat.value}</p>
              </div>
              <div className={`p-3 rounded-lg ${stat.bg}`}>
                <Icon className={`h-6 w-6 ${stat.color}`} />
              </div>
            </div>
          </Card>
        );
      })}
    </div>
  );
};

export default StatsCards;