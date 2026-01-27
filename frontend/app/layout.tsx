export const metadata = {
  title: "AI Underwriter Assistant (Lab 8)",
  description: "InsurTech RAG demo"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="th">
      <body style={{ margin: 0, fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
