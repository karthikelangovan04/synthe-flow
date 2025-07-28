import { Header } from "@/components/Header";
import { Hero } from "@/components/Hero";
import { UseCases } from "@/components/UseCases";
import { Architecture } from "@/components/Architecture";
import { Footer } from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <Hero />
      <UseCases />
      <Architecture />
      <Footer />
    </div>
  );
};

export default Index;
