import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Waste Classification System | AI-Powered Waste Recognition",
  description: "Intelligent waste material classification using a hierarchical CNN. Identify 8 waste categories including cardboard, glass, metal, plastic, organic, paper, textile, and battery."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
