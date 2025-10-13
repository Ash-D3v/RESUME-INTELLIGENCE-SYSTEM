import { cn } from "@/lib/utils";

interface ProgressCircleProps {
  value: number;
  size?: number;
  strokeWidth?: number;
  showLabel?: boolean;
  variant?: "default" | "success" | "warning" | "destructive";
}

export default function ProgressCircle({
  value,
  size = 120,
  strokeWidth = 8,
  showLabel = true,
  variant = "default",
}: ProgressCircleProps) {
  const normalizedValue = Math.min(100, Math.max(0, value));
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (normalizedValue / 100) * circumference;

  const getColor = () => {
    if (variant === "success") return "hsl(var(--success))";
    if (variant === "warning") return "hsl(var(--warning))";
    if (variant === "destructive") return "hsl(var(--destructive))";
    
    if (normalizedValue >= 85) return "hsl(var(--success))";
    if (normalizedValue >= 70) return "hsl(var(--warning))";
    return "hsl(var(--destructive))";
  };

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="hsl(var(--muted))"
          strokeWidth={strokeWidth}
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={getColor()}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
          style={{
            filter: "drop-shadow(0 0 8px currentColor)",
          }}
        />
      </svg>
      {showLabel && (
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold" style={{ color: getColor() }}>
            {normalizedValue}
          </span>
          <span className="text-xs text-muted-foreground font-medium">Score</span>
        </div>
      )}
    </div>
  );
}
