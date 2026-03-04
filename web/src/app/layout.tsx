import type { Metadata } from "next";
import { Manrope, Source_Serif_4 } from "next/font/google";
import "./globals.css";
import RevealObserver from "@/components/reveal-observer";

const manrope = Manrope({ subsets: ["latin"], variable: "--font-manrope" });
const sourceSerif = Source_Serif_4({
  subsets: ["latin"],
  weight: ["300", "400", "600", "700"],
  variable: "--font-source-serif",
});

export const metadata: Metadata = {
  title: "MergeCraft — Contributions that get merged.",
  description:
    "Trained on merged vs. rejected PR pairs across 1,000 repos — the first model that understands the unwritten social rules of open source, not just code quality.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${manrope.variable} ${sourceSerif.variable}`}>
      <body style={{ fontFamily: "var(--font-manrope), system-ui, sans-serif" }}>
        <RevealObserver />{children}
      </body>
    </html>
  );
}
