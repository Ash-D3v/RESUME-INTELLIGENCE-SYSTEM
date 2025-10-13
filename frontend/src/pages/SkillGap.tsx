import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import SkillBadge from "@/components/SkillBadge";
import ProgressCircle from "@/components/ProgressCircle";
import { Plus, X, Sparkles, TrendingUp } from "lucide-react";

export default function SkillGap() {
  const [userSkills, setUserSkills] = useState<string[]>(["Python", "React", "SQL"]);
  const [requiredSkills, setRequiredSkills] = useState<string[]>(["Python", "React", "TypeScript", "AWS", "Docker"]);
  const [newSkill, setNewSkill] = useState("");

  const addUserSkill = () => {
    if (newSkill.trim()) {
      setUserSkills([...userSkills, newSkill.trim()]);
      setNewSkill("");
    }
  };

  const removeUserSkill = (skill: string) => {
    setUserSkills(userSkills.filter((s) => s !== skill));
  };

  const matchingSkills = userSkills.filter((skill) =>
    requiredSkills.some((req) => req.toLowerCase() === skill.toLowerCase())
  );

  const missingSkills = requiredSkills.filter(
    (skill) => !userSkills.some((user) => user.toLowerCase() === skill.toLowerCase())
  );

  const matchPercentage = requiredSkills.length > 0
    ? Math.round((matchingSkills.length / requiredSkills.length) * 100)
    : 0;

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="space-y-4">
        <h1 className="text-4xl font-bold">Skill Gap Analysis</h1>
        <p className="text-lg text-muted-foreground">
          Compare your skills against job requirements and identify areas for growth
        </p>
      </div>

      {/* Input Section */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* User Skills */}
        <Card className="glass-card border shadow-lg">
          <CardHeader>
            <CardTitle>Your Skills</CardTitle>
            <CardDescription>Add skills from your resume or expertise</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <Input
                placeholder="Enter a skill..."
                value={newSkill}
                onChange={(e) => setNewSkill(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && addUserSkill()}
              />
              <Button onClick={addUserSkill} size="icon" className="shadow-md">
                <Plus className="w-4 h-4" />
              </Button>
            </div>
            <div className="flex flex-wrap gap-2 min-h-[100px] p-4 rounded-lg border bg-muted/30">
              {userSkills.map((skill) => (
                <div key={skill} className="flex items-center gap-1 bg-card border rounded-lg px-3 py-1.5">
                  <span className="font-medium text-sm">{skill}</span>
                  <button
                    onClick={() => removeUserSkill(skill)}
                    className="ml-1 text-muted-foreground hover:text-destructive transition-colors"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Required Skills */}
        <Card className="glass-card border shadow-lg">
          <CardHeader>
            <CardTitle>Job Requirements</CardTitle>
            <CardDescription>Skills needed for your target role</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2 min-h-[100px] p-4 rounded-lg border bg-muted/30">
              {requiredSkills.map((skill) => (
                <SkillBadge
                  key={skill}
                  skill={skill}
                  variant={
                    matchingSkills.some((s) => s.toLowerCase() === skill.toLowerCase())
                      ? "success"
                      : "destructive"
                  }
                />
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Analysis Results */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Match Score */}
        <Card className="glass-card border shadow-lg">
          <CardHeader>
            <CardTitle>Match Score</CardTitle>
            <CardDescription>Overall skill compatibility</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center gap-4">
            <ProgressCircle value={matchPercentage} size={140} />
            <p className="text-sm text-center text-muted-foreground">
              You have {matchingSkills.length} of {requiredSkills.length} required skills
            </p>
          </CardContent>
        </Card>

        {/* Matching Skills */}
        <Card className="glass-card border shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-success" />
              Matching Skills
            </CardTitle>
            <CardDescription>Skills you already have</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {matchingSkills.length > 0 ? (
                matchingSkills.map((skill) => (
                  <div
                    key={skill}
                    className="flex items-center gap-3 p-3 rounded-lg bg-success/10 border border-success/20"
                  >
                    <div className="w-2 h-2 rounded-full bg-success" />
                    <span className="font-medium text-success">{skill}</span>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">
                  No matching skills found
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Missing Skills */}
        <Card className="glass-card border shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-warning" />
              Skills to Learn
            </CardTitle>
            <CardDescription>Areas for improvement</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {missingSkills.length > 0 ? (
                missingSkills.map((skill) => (
                  <div
                    key={skill}
                    className="flex items-center justify-between p-3 rounded-lg bg-destructive/10 border border-destructive/20"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 rounded-full bg-destructive" />
                      <span className="font-medium text-destructive">{skill}</span>
                    </div>
                    <SkillBadge skill="Priority" variant="warning" />
                  </div>
                ))
              ) : (
                <p className="text-sm text-success text-center py-8 font-medium">
                  🎉 You have all required skills!
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recommendations */}
      {missingSkills.length > 0 && (
        <Card className="glass-card border shadow-lg">
          <CardHeader>
            <CardTitle>Learning Path Recommendations</CardTitle>
            <CardDescription>Suggested courses to bridge your skill gaps</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {missingSkills.slice(0, 3).map((skill) => (
                <div key={skill} className="p-4 rounded-lg border bg-card/50 hover:bg-card transition-colors">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-semibold">Master {skill}</h4>
                      <p className="text-sm text-muted-foreground mt-1">
                        Complete {skill} course • Beginner to Advanced • 20 hours
                      </p>
                      <div className="flex gap-2 mt-3">
                        <SkillBadge skill="Coursera" variant="default" />
                        <SkillBadge skill="4.8 ⭐" variant="success" />
                      </div>
                    </div>
                    <Button size="sm" className="shadow-md">
                      Enroll
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
