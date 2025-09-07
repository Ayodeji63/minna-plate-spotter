import { Search, Calendar, RotateCcw } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface SearchFiltersProps {
  searchTerm: string;
  onSearchChange: (value: string) => void;
  dateFrom: string;
  onDateFromChange: (value: string) => void;
  dateTo: string;
  onDateToChange: (value: string) => void;
  onClearFilters: () => void;
}

const SearchFilters = ({
  searchTerm,
  onSearchChange,
  dateFrom,
  onDateFromChange,
  dateTo,
  onDateToChange,
  onClearFilters,
}: SearchFiltersProps) => {
  return (
    <Card className="p-6 bg-gradient-card border-nigeria-green/20 shadow-card">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Search Box */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search plate numbers..."
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-10 border-nigeria-green/30 focus:ring-nigeria-green focus:border-nigeria-green"
          />
        </div>

        {/* Date From */}
        <div className="relative">
          <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="datetime-local"
            value={dateFrom}
            onChange={(e) => onDateFromChange(e.target.value)}
            className="pl-10 border-nigeria-green/30 focus:ring-nigeria-green focus:border-nigeria-green"
          />
        </div>

        {/* Date To */}
        <div className="relative">
          <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="datetime-local"
            value={dateTo}
            onChange={(e) => onDateToChange(e.target.value)}
            className="pl-10 border-nigeria-green/30 focus:ring-nigeria-green focus:border-nigeria-green"
          />
        </div>

        {/* Clear Filters */}
        <Button
          onClick={onClearFilters}
          variant="outline"
          className="border-nigeria-green text-nigeria-green hover:bg-nigeria-green hover:text-white"
        >
          <RotateCcw className="h-4 w-4 mr-2" />
          Clear Filters
        </Button>
      </div>
    </Card>
  );
};

export default SearchFilters;