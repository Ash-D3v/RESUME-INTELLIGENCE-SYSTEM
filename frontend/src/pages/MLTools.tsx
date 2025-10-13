import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ProgressCircle from "@/components/ProgressCircle";
import SkillBadge from "@/components/SkillBadge";
import { Brain, Zap, Users, Sparkles } from "lucide-react";

export default function MLTools() {
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(false);

  const handleAnalyze = () => {
    setAnalyzing(true);
    setTimeout(() => {
      setAnalyzing(false);
      setResults(true);
    }, 1500);
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="space-y-4">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20">
          <Brain className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-primary">Advanced ML Analytics</span>
        </div>
        <h1 className="text-4xl font-bold">ML-Powered Analysis Tools</h1>
        <p className="text-lg text-muted-foreground">
          Leverage machine learning for semantic skill matching, job fit prediction, and personalized recommendations
        </p>
      </div>

      <Tabs defaultValue="matching" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="matching">
            <Sparkles className="w-4 h-4 mr-2" />
            Skill Matching
          </TabsTrigger>
          <TabsTrigger value="prediction">
            <Zap className="w-4 h-4 mr-2" />
            Job Fit
          </TabsTrigger>
          <TabsTrigger value="collaborative">
            <Users className="w-4 h-4 mr-2" />
            Recommendations
          </TabsTrigger>
        </TabsList>

        {/* Semantic Skill Matching */}
        <TabsContent value="matching" className="space-y-6 mt-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>Your Skills</CardTitle>
                <CardDescription>Enter your technical skills</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Input placeholder="e.g., Python, Machine Learning" />
                <Input placeholder="e.g., React, TypeScript" />
                <Input placeholder="e.g., Docker, AWS" />
              </CardContent>
            </Card>

            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>Job Requirements</CardTitle>
                <CardDescription>Skills from job posting</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Input placeholder="e.g., Python, Deep Learning" />
                <Input placeholder="e.g., React.js, JavaScript" />
                <Input placeholder="e.g., Kubernetes, Cloud" />
              </CardContent>
            </Card>
          </div>

          <Button
            onClick={handleAnalyze}
            disabled={analyzing}
            className="w-full shadow-lg h-12 text-base"
          >
            {analyzing ? (
              <>
                <Brain className="w-5 h-5 mr-2 animate-spin" />
                Analyzing with Word2Vec...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 mr-2" />
                Analyze Semantic Match
              </>
            )}
          </Button>

          {results && (
            <div className="grid gap-6 lg:grid-cols-3 animate-scale-in">
              <Card className="glass-card border shadow-lg">
                <CardHeader>
                  <CardTitle>Exact Matches</CardTitle>
                  <CardDescription>Direct skill alignment</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  <SkillBadge skill="Python" variant="success" />
                  <SkillBadge skill="React" variant="success" />
                </CardContent>
              </Card>

              <Card className="glass-card border shadow-lg">
                <CardHeader>
                  <CardTitle>Semantic Matches</CardTitle>
                  <CardDescription>Similar skills (Word2Vec)</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                    <span className="text-sm font-medium">Machine Learning → Deep Learning</span>
                    <SkillBadge skill="92%" variant="success" />
                  </div>
                  <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                    <span className="text-sm font-medium">Docker → Kubernetes</span>
                    <SkillBadge skill="85%" variant="success" />
                  </div>
                </CardContent>
              </Card>

              <Card className="glass-card border shadow-lg">
                <CardHeader>
                  <CardTitle>Overall Score</CardTitle>
                  <CardDescription>Combined TF-IDF + Semantic</CardDescription>
                </CardHeader>
                <CardContent className="flex justify-center">
                  <ProgressCircle value={88} size={120} variant="success" />
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* Job Fit Prediction */}
        <TabsContent value="prediction" className="space-y-6 mt-6">
          <Card className="glass-card border shadow-lg">
            <CardHeader>
              <CardTitle>Job Fit Predictor</CardTitle>
              <CardDescription>
                Random Forest classifier predicts your fit probability for target roles
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-sm font-medium mb-2 block">Your Skills</label>
                  <textarea
                    className="w-full h-24 p-3 rounded-lg border bg-background resize-none"
                    placeholder="Enter comma-separated skills..."
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Target Job Skills</label>
                  <textarea
                    className="w-full h-24 p-3 rounded-lg border bg-background resize-none"
                    placeholder="Enter required skills..."
                  />
                </div>
              </div>

              <Button onClick={handleAnalyze} disabled={analyzing} className="w-full shadow-lg h-12">
                {analyzing ? (
                  <>
                    <Brain className="w-5 h-5 mr-2 animate-spin" />
                    Running ML Model...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5 mr-2" />
                    Predict Job Fit
                  </>
                )}
              </Button>

              {results && (
                <div className="grid gap-6 md:grid-cols-2 animate-scale-in">
                  <Card className="bg-gradient-to-br from-success/10 to-success/5 border-success/20">
                    <CardHeader>
                      <CardTitle className="text-success">ML Prediction</CardTitle>
                      <CardDescription>Random Forest probability</CardDescription>
                    </CardHeader>
                    <CardContent className="flex flex-col items-center gap-3">
                      <div className="text-5xl font-bold text-success">84%</div>
                      <SkillBadge skill="High Fit" variant="success" />
                      <p className="text-sm text-center text-muted-foreground">
                        Strong match based on skill patterns
                      </p>
                    </CardContent>
                  </Card>

                  <Card className="glass-card border shadow-lg">
                    <CardHeader>
                      <CardTitle>Confidence Analysis</CardTitle>
                      <CardDescription>Model certainty breakdown</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Model Accuracy</span>
                        <span className="font-bold">92%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Feature Importance</span>
                        <SkillBadge skill="View" variant="default" />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Recommendation</span>
                        <SkillBadge skill="Apply" variant="success" />
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Collaborative Filtering */}
        <TabsContent value="collaborative" className="space-y-6 mt-6">
          <Card className="glass-card border shadow-lg">
            <CardHeader>
              <CardTitle>Personalized Job Recommendations</CardTitle>
              <CardDescription>
                Collaborative filtering based on user interaction patterns
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="p-4 rounded-lg bg-muted/30 border">
                <p className="text-sm text-muted-foreground mb-2">Interaction History</p>
                <div className="flex gap-2 flex-wrap">
                  <SkillBadge skill="5 jobs viewed" variant="default" />
                  <SkillBadge skill="3 jobs saved" variant="success" />
                  <SkillBadge skill="2 applications" variant="warning" />
                </div>
              </div>

              <Button onClick={handleAnalyze} disabled={analyzing} className="w-full shadow-lg h-12">
                {analyzing ? (
                  <>
                    <Brain className="w-5 h-5 mr-2 animate-spin" />
                    Generating Recommendations...
                  </>
                ) : (
                  <>
                    <Users className="w-5 h-5 mr-2" />
                    Get Personalized Jobs
                  </>
                )}
              </Button>

              {results && (
                <div className="space-y-4 animate-slide-up">
                  {[1, 2, 3].map((i) => (
                    <Card key={i} className="bg-card/50 border hover:bg-card transition-colors">
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-semibold">Senior ML Engineer</h4>
                            <p className="text-sm text-muted-foreground mt-1">
                              Similar users with 85% match applied to this role
                            </p>
                            <div className="flex gap-2 mt-3">
                              <SkillBadge skill="Python" variant="success" />
                              <SkillBadge skill="TensorFlow" variant="success" />
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-2xl font-bold text-primary">92%</div>
                            <p className="text-xs text-muted-foreground">Similarity</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
