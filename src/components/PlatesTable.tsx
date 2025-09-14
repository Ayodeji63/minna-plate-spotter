import { format } from 'date-fns';
import { MapPin, Clock, Camera, AlertCircle } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { PlateRecord } from '@/lib/supabase.ts';

interface PlatesTableProps {
  plates: PlateRecord[];
  loading: boolean;
}

const PlatesTable = ({ plates, loading }: PlatesTableProps) => {
  const formatNigerianTime = (dateString: string) => {
    return format(new Date(dateString), 'MMM dd, yyyy - HH:mm:ss');
  };

  if (loading) {
    return (
      <Card className="p-8 text-center bg-gradient-card shadow-card">
        <div className="animate-spin h-8 w-8 border-4 border-nigeria-green border-t-transparent rounded-full mx-auto mb-4"></div>
        <p className="text-muted-foreground">Loading plate records...</p>
      </Card>
    );
  }

  if (plates.length === 0) {
    return (
      <Card className="p-8 text-center bg-gradient-card shadow-card">
        <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">No Records Found</h3>
        <p className="text-muted-foreground">
          No license plates have been detected yet. The system will update automatically when new plates are captured.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Desktop Table */}
      <div className="hidden md:block">
        <Card className="overflow-hidden bg-gradient-card shadow-card">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-nigeria-green text-white">
                  <th className="text-left py-4 px-6 font-semibold">Plate Number</th>
                  <th className="text-left py-4 px-6 font-semibold">Date & Time</th>
                  <th className="text-left py-4 px-6 font-semibold">Junction</th>
                  <th className="text-left py-4 px-6 font-semibold">Image</th>
                </tr>
              </thead>
              <tbody>
                {plates.map((plate, index) => (
                  <tr
                    key={plate.id}
                    className={`border-b border-nigeria-green/10 hover:bg-nigeria-green/5 transition-colors ${
                      index % 2 === 0 ? 'bg-white/50' : 'bg-nigeria-green/5'
                    }`}
                  >
                    <td className="py-4 px-6">
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="bg-nigeria-green text-white font-mono text-sm">
                          {plate.plate_text}
                        </Badge>
                      </div>
                    </td>
                    <td className="py-4 px-6">
                      <div className="flex items-center gap-2 text-sm text-foreground">
                        <Clock className="h-4 w-4 text-muted-foreground" />
                        {formatNigerianTime(plate.captured_at)}
                      </div>
                    </td>
                    <td className="py-4 px-6">
                      <div className="flex items-center gap-2 text-sm text-foreground">
                        <MapPin className="h-4 w-4 text-nigeria-green" />
                        {plate.junction}
                      </div>
                    </td>
                    <td className="py-4 px-6">
                      {plate.image_url ? (
                        <img
                          src={plate.image_url}
                          alt={`Plate ${plate.plate_text}`}
                          className="h-12 w-16 object-cover rounded-md border border-nigeria-green/20"
                        />
                      ) : (
                        <div className="h-12 w-16 bg-muted rounded-md flex items-center justify-center">
                          <Camera className="h-4 w-4 text-muted-foreground" />
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* Mobile Cards */}
      <div className="md:hidden space-y-3">
        {plates.map((plate) => (
          <Card key={plate.id} className="p-4 bg-gradient-card shadow-card hover:shadow-hover transition-shadow">
            <div className="flex items-start justify-between mb-3">
              <Badge variant="secondary" className="bg-nigeria-green text-white font-mono">
                {plate.plate_text}
              </Badge>
              {plate.image_url && (
                <img
                  src={plate.image_url}
                  alt={`Plate ${plate.plate_text}`}
                  className="h-12 w-16 object-cover rounded-md border border-nigeria-green/20"
                />
              )}
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm text-foreground">
                <Clock className="h-4 w-4 text-muted-foreground" />
                {formatNigerianTime(plate.captured_at)}
              </div>
              <div className="flex items-center gap-2 text-sm text-foreground">
                <MapPin className="h-4 w-4 text-nigeria-green" />
                {plate.junction}
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default PlatesTable;