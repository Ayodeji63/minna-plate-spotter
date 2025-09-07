import { useState, useMemo } from 'react';
import Header from '@/components/Header';
import SearchFilters from '@/components/SearchFilters';
import StatsCards from '@/components/StatsCards';
import PlatesTable from '@/components/PlatesTable';
import { usePlates } from '@/hooks/usePlates';

const Index = () => {
  const { plates, loading } = usePlates();
  const [searchTerm, setSearchTerm] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');

  // Filter plates based on search criteria
  const filteredPlates = useMemo(() => {
    return plates.filter((plate) => {
      const matchesSearch = !searchTerm || 
        plate.plate_text.toLowerCase().includes(searchTerm.toLowerCase()) ||
        plate.junction.toLowerCase().includes(searchTerm.toLowerCase());

      const plateDate = new Date(plate.captured_at);
      const matchesDateFrom = !dateFrom || plateDate >= new Date(dateFrom);
      const matchesDateTo = !dateTo || plateDate <= new Date(dateTo);

      return matchesSearch && matchesDateFrom && matchesDateTo;
    });
  }, [plates, searchTerm, dateFrom, dateTo]);

  const handleClearFilters = () => {
    setSearchTerm('');
    setDateFrom('');
    setDateTo('');
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <StatsCards plates={plates} />
        
        <SearchFilters
          searchTerm={searchTerm}
          onSearchChange={setSearchTerm}
          dateFrom={dateFrom}
          onDateFromChange={setDateFrom}
          dateTo={dateTo}
          onDateToChange={setDateTo}
          onClearFilters={handleClearFilters}
        />
        
        <div className="mt-6">
          <PlatesTable plates={filteredPlates} loading={loading} />
        </div>
      </main>
    </div>
  );
};

export default Index;
