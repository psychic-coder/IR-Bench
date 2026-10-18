"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { ThemeToggle } from "@/components/ThemeToggle";

const LINKS = [
  { href: "/search", label: "Search" },
  { href: "/compare", label: "Compare" },
  { href: "/benchmark", label: "Benchmark" },
  { href: "/hard-queries", label: "Hard queries" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <header className="border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-950/80">
      <nav
        className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3"
        aria-label="Main"
      >
        <Link href="/search" className="font-semibold tracking-tight">
          irbench <span className="font-normal text-slate-500">retrieval benchmark</span>
        </Link>
        <ul className="flex flex-wrap gap-1">
          {LINKS.map((l) => {
            const active = pathname?.startsWith(l.href);
            return (
              <li key={l.href}>
                <Link
                  href={l.href}
                  aria-current={active ? "page" : undefined}
                  className={`rounded px-3 py-1.5 text-sm ${
                    active
                      ? "bg-accent-soft text-accent dark:bg-blue-950 dark:text-blue-300"
                      : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
                  }`}
                >
                  {l.label}
                </Link>
              </li>
            );
          })}
        </ul>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </nav>
    </header>
  );
}
