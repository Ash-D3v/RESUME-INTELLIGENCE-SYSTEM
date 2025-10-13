import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface SkillBadgeProps {
  skill: string;
  level?: "beginner" | "intermediate" | "advanced" | "expert";
  variant?: "default" | "success" | "warning" | "destructive";
}

export default function SkillBadge({ skill, level, variant = "default" }: SkillBadgeProps) {
  const levelColors = {
    beginner: "bg-blue-500/10 text-blue-600 border-blue-500/20",
    intermediate: "bg-green-500/10 text-green-600 border-green-500/20",
    advanced: "bg-purple-500/10 text-purple-600 border-purple-500/20",
    expert: "bg-orange-500/10 text-orange-600 border-orange-500/20",
  };

  const variantStyles = {
    default: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
    success: "bg-success/10 text-success border-success/20 hover:bg-success/20",
    warning: "bg-warning/10 text-warning border-warning/20 hover:bg-warning/20",
    destructive: "bg-destructive/10 text-destructive border-destructive/20 hover:bg-destructive/20",
  };

  return (
    <Badge
      className={cn(
        "px-3 py-1.5 font-medium transition-all hover:scale-105",
        level ? levelColors[level] : variantStyles[variant]
      )}
    >
      {skill}
      {level && <span className="ml-1 text-xs opacity-70">({level})</span>}
    </Badge>
  );
}
