import { Car, MapPin } from 'lucide-react';

const Header = () => {
  return (
    <header className="bg-gradient-hero text-white shadow-lg">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-white/20 rounded-lg">
            <Car className="h-8 w-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Vehicle Recognition</h1>
            <div className="flex items-center gap-2 text-white/90">
              <MapPin className="h-4 w-4" />
              <span className="text-lg">Minna, Niger State</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;