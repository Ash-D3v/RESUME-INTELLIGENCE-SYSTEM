import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import StatCard from "@/components/StatCard";
import ProgressCircle from "@/components/ProgressCircle";
import SkillBadge from "@/components/SkillBadge";
import { TrendingUp, Users, Briefcase, Target, Brain, Zap, Network } from "lucide-react";

export default function Analytics() {
  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="space-y-4">
        <h1 className="text-4xl font-bold">Analytics Dashboard</h1>
        <p className="text-lg text-muted-foreground">
          Insights into resume trends, job market, and ML model performance
        </p>
      </div>

      <Tabs defaultValue="resume" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="resume">Resume Analytics</TabsTrigger>
          <TabsTrigger value="market">Market Insights</TabsTrigger>
          <TabsTrigger value="ml">ML Performance</TabsTrigger>
        </TabsList>

        {/* Resume Analytics */}
        <TabsContent value="resume" className="space-y-6 mt-6">
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            <StatCard
              title="Total Resumes"
              value="1,247"
              icon={Briefcase}
              variant="primary"
              trend={{ value: 12, isPositive: true }}
            />
            <StatCard
              title="Avg ATS Score"
              value="76"
              icon={Target}
              variant="success"
              trend={{ value: 5, isPositive: true }}
            />
            <StatCard
              title="Skills Analyzed"
              value="8,934"
              icon={Zap}
              variant="default"
              trend={{ value: 18, isPositive: true }}
            />
            <StatCard
              title="Active Users"
              value="892"
              icon={Users}
              variant="warning"
              trend={{ value: 8, isPositive: true }}
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>ATS Score Distribution</CardTitle>
                <CardDescription>Breakdown of resume scores</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Excellent (85-100)</span>
                    <div className="flex items-center gap-3">
                      <div className="w-48 h-3 bg-muted rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-success to-success/70 rounded-full" style={{ width: "35%" }} />
                      </div>
                      <span className="text-sm font-bold w-12 text-right">35%</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Good (70-84)</span>
                    <div className="flex items-center gap-3">
                      <div className="w-48 h-3 bg-muted rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-warning to-warning/70 rounded-full" style={{ width: "42%" }} />
                      </div>
                      <span className="text-sm font-bold w-12 text-right">42%</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Needs Work (&lt;70)</span>
                    <div className="flex items-center gap-3">
                      <div className="w-48 h-3 bg-muted rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-destructive to-destructive/70 rounded-full" style={{ width: "23%" }} />
                      </div>
                      <span className="text-sm font-bold w-12 text-right">23%</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>Common Skill Gaps</CardTitle>
                <CardDescription>Most frequently missing skills</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { skill: "Cloud Computing", percentage: 67 },
                    { skill: "DevOps", percentage: 54 },
                    { skill: "Machine Learning", percentage: 48 },
                    { skill: "Kubernetes", percentage: 42 },
                    { skill: "TypeScript", percentage: 38 },
                  ].map((item) => (
                    <div key={item.skill} className="flex items-center justify-between p-3 rounded-lg bg-muted/30">
                      <span className="font-medium">{item.skill}</span>
                      <SkillBadge skill={`${item.percentage}%`} variant="warning" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Market Insights */}
        <TabsContent value="market" className="space-y-6 mt-6">
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            <StatCard
              title="Active Jobs"
              value="12,458"
              icon={Briefcase}
              variant="primary"
              trend={{ value: 15, isPositive: true }}
            />
            <StatCard
              title="Avg Match Rate"
              value="68%"
              icon={Target}
              variant="success"
            />
            <StatCard
              title="Hot Skills"
              value="24"
              icon={TrendingUp}
              variant="warning"
            />
            <StatCard
              title="Companies"
              value="3,247"
              icon={Users}
              variant="default"
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>Most In-Demand Skills</CardTitle>
                <CardDescription>Based on job postings analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { skill: "Python", demand: 92, growth: 15 },
                    { skill: "React", demand: 88, growth: 22 },
                    { skill: "AWS", demand: 85, growth: 18 },
                    { skill: "Docker", demand: 78, growth: 28 },
                    { skill: "TypeScript", demand: 75, growth: 35 },
                  ].map((item) => (
                    <div key={item.skill} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">{item.skill}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-success">+{item.growth}%</span>
                          <span className="text-sm font-bold">{item.demand}%</span>
                        </div>
                      </div>
                      <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full gradient-primary rounded-full transition-all"
                          style={{ width: `${item.demand}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>Salary Trends</CardTitle>
                <CardDescription>Average salaries by role</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { role: "ML Engineer", salary: "$165k", trend: 12 },
                    { role: "Full Stack Dev", salary: "$142k", trend: 8 },
                    { role: "Data Scientist", salary: "$138k", trend: 10 },
                    { role: "DevOps Engineer", salary: "$135k", trend: 15 },
                    { role: "Backend Dev", salary: "$128k", trend: 6 },
                  ].map((item) => (
                    <div key={item.role} className="flex items-center justify-between p-3 rounded-lg bg-muted/30">
                      <div className="flex-1">
                        <p className="font-semibold">{item.role}</p>
                        <p className="text-sm text-muted-foreground">+{item.trend}% YoY</p>
                      </div>
                      <span className="text-lg font-bold text-primary">{item.salary}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* ML Performance */}
        <TabsContent value="ml" className="space-y-6 mt-6">
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            <StatCard
              title="Model Accuracy"
              value="92%"
              icon={Brain}
              variant="success"
            />
            <StatCard
              title="Predictions Made"
              value="5,438"
              icon={Zap}
              variant="primary"
            />
            <StatCard
              title="Vocab Size"
              value="12.5k"
              icon={Network}
              variant="default"
            />
            <StatCard
              title="Avg Response"
              value="124ms"
              icon={TrendingUp}
              variant="warning"
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>Word2Vec Model</CardTitle>
                <CardDescription>Semantic similarity engine</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-center">
                  <ProgressCircle value={94} size={120} variant="success" />
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Vocabulary</span>
                    <span className="font-semibold">12,487 terms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Vector Dim</span>
                    <span className="font-semibold">100</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Training Data</span>
                    <span className="font-semibold">2.1M resumes</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>Random Forest</CardTitle>
                <CardDescription>Job fit prediction model</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-center">
                  <ProgressCircle value={89} size={120} variant="warning" />
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Estimators</span>
                    <span className="font-semibold">100 trees</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Features</span>
                    <span className="font-semibold">45 skills</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">F1 Score</span>
                    <span className="font-semibold">0.91</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>Clustering</CardTitle>
                <CardDescription>User segmentation</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-center">
                  <ProgressCircle value={87} size={120} variant="default" />
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Clusters</span>
                    <span className="font-semibold">5 segments</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Silhouette</span>
                    <span className="font-semibold">0.74</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Users</span>
                    <span className="font-semibold">8,492</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="glass-card border shadow-lg">
            <CardHeader>
              <CardTitle>Feature Importance</CardTitle>
              <CardDescription>Top predictive features in job fit model</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { feature: "Years of Experience", importance: 0.28 },
                  { feature: "Technical Skills Match", importance: 0.24 },
                  { feature: "Education Level", importance: 0.18 },
                  { feature: "Industry Experience", importance: 0.15 },
                  { feature: "Certification Count", importance: 0.15 },
                ].map((item) => (
                  <div key={item.feature} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{item.feature}</span>
                      <span className="text-sm font-bold">{(item.importance * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full h-2.5 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full gradient-accent rounded-full transition-all"
                        style={{ width: `${item.importance * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
