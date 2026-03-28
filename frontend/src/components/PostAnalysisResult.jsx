import React, { useState, useEffect } from 'react';
import { AlertTriangle, Brain, Download, Loader2 } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

// Helper: Convert decimal (0-1) to percentage string
const toPercent = (val) => Math.round((val || 0) * 100);

// Helper: Classify risk level from score
const getRiskLevel = (score) => {
  if (score < 0.3) return 'low';
  if (score < 0.6) return 'medium';
  return 'high';
};

// Helper: Get risk badge config
const getRiskBadge = (level) => {
  const badges = {
    low: { bg: 'bg-green-50 border-green-100 text-green-600', text: 'LOW RISK' },
    medium: { bg: 'bg-amber-50 border-amber-100 text-amber-600', text: 'MEDIUM RISK' },
    high: { bg: 'bg-red-50 border-red-100 text-red-600', text: 'HIGH RISK' },
  };
  return badges[level] || { bg: 'bg-gray-50 border-gray-200 text-gray-500', text: 'PENDING' };
};

export default function PostAnalysisResult({
  fusionScore = 0,
  videoScore = 0,
  questionnaireScore = 0,
  videoDetails = {},
  questionnaireDetails = {},
  patientName = 'Child',
}) {
  const [aiSummary, setAiSummary] = useState('');
  const [aiExplanation, setAiExplanation] = useState('');
  const [aiGuidance, setAiGuidance] = useState('');
  const [loading, setLoading] = useState(true);

  // Convert decimals to percentages for display
  const fusionPercent = toPercent(fusionScore);
  const videoPercent = toPercent(videoScore);
  const questionnairePercent = toPercent(questionnaireScore);
  const riskLevel = getRiskLevel(fusionScore);
  const riskBadge = getRiskBadge(riskLevel);

  // Video details (all decimals 0-1)
  const headBanging = toPercent(videoDetails.headBanging || 0);
  const armFlapping = toPercent(videoDetails.armFlapping || 0);
  const spinning = toPercent(videoDetails.spinning || 0);

  // Questionnaire details (all decimals 0-1)
  const social = toPercent(questionnaireDetails.social || 0);
  const communication = toPercent(questionnaireDetails.communication || 0);
  const behavioral = toPercent(questionnaireDetails.behavioral || 0);
  const emotional = toPercent(questionnaireDetails.emotional || 0);

  // Chart data
  const chartData = [
    { name: 'Risk', value: fusionPercent },
    { name: 'Safe', value: 100 - fusionPercent },
  ];
  const riskColors = { low: '#22c55e', medium: '#f59e0b', high: '#ef4444' };
  const COLORS = [riskColors[riskLevel] || '#6b7280', '#f3f4f6'];

  // Fetch all AI content in parallel
  useEffect(() => {
    const fetchAIContent = async () => {
      const apiKey = import.meta.env.VITE_GROQ_API_KEY;
      
      // Fallback texts
      const fallbackSummary = 'This is a risk assessment, NOT a clinical diagnosis. Please share these results with a qualified healthcare professional.';
      const fallbackExplanation = 'The screening results indicate patterns that warrant further discussion with a healthcare provider. Video analysis and questionnaire responses have been combined to generate an overall risk assessment.';
      const fallbackGuidance = 'We recommend consulting with a pediatrician or developmental specialist to discuss these results. Early intervention, when needed, can make a significant positive impact.';

      if (!apiKey) {
        setAiSummary(fallbackSummary);
        setAiExplanation(fallbackExplanation);
        setAiGuidance(fallbackGuidance);
        setLoading(false);
        return;
      }

      setLoading(true);

      try {
        // Build prompts
        const summaryPrompt = `Generate a 2-3 sentence empathetic summary for a parent whose child (${patientName}) completed an autism screening. The overall risk score is ${fusionPercent}% (${riskLevel} risk). Be supportive and non-alarming. Do not mention specific numbers.`;

        const explanationPrompt = `Generate a 4-5 sentence detailed explanation of screening results for a parent. The fusion score is ${fusionPercent}% (${riskLevel} risk). Video analysis detected: arm flapping ${armFlapping}%, spinning ${spinning}%, head banging ${headBanging}%. Questionnaire scores: social ${social}%, communication ${communication}%, behavioral ${behavioral}%, emotional ${emotional}%. Explain what these indicators mean in plain language. Be informative but not diagnostic.`;

        const guidancePrompt = `Generate 3-4 actionable next steps for parents whose child (${patientName}) received a ${riskLevel} risk autism screening result (${fusionPercent}%). Include recommendations about professional consultation, observation, and support. Be encouraging and practical.`;

        // Make all three API calls in parallel
        const [summaryRes, explanationRes, guidanceRes] = await Promise.all([
          fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
              model: 'llama-3.3-70b-versatile',
              messages: [{ role: 'user', content: summaryPrompt }],
              max_tokens: 200,
              temperature: 0.7
            })
          }),
          fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
              model: 'llama-3.3-70b-versatile',
              messages: [{ role: 'user', content: explanationPrompt }],
              max_tokens: 300,
              temperature: 0.7
            })
          }),
          fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
              model: 'llama-3.3-70b-versatile',
              messages: [{ role: 'user', content: guidancePrompt }],
              max_tokens: 250,
              temperature: 0.7
            })
          })
        ]);

        // Parse responses
        if (summaryRes.ok) {
          const data = await summaryRes.json();
          setAiSummary(data.choices[0].message.content);
        } else {
          setAiSummary(fallbackSummary);
        }

        if (explanationRes.ok) {
          const data = await explanationRes.json();
          setAiExplanation(data.choices[0].message.content);
        } else {
          setAiExplanation(fallbackExplanation);
        }

        if (guidanceRes.ok) {
          const data = await guidanceRes.json();
          setAiGuidance(data.choices[0].message.content);
        } else {
          setAiGuidance(fallbackGuidance);
        }
      } catch (err) {
        console.error('AI content generation failed:', err);
        setAiSummary(fallbackSummary);
        setAiExplanation(fallbackExplanation);
        setAiGuidance(fallbackGuidance);
      } finally {
        setLoading(false);
      }
    };

    fetchAIContent();
  }, [fusionScore, riskLevel, patientName, armFlapping, spinning, headBanging, social, communication, behavioral, emotional, fusionPercent]);

  // PDF Download handler
  const handleDownloadPDF = () => {
    const content = `
AUTISM SCREENING REPORT
=======================
Patient: ${patientName}
Date: ${new Date().toLocaleDateString()}

SUMMARY
-------
${aiSummary}

OVERALL RISK ASSESSMENT
-----------------------
Risk Level: ${riskBadge.text}
Fusion Score: ${fusionPercent}%
Video Score: ${videoPercent}%
Questionnaire Score: ${questionnairePercent}%

VIDEO ANALYSIS DETAILS
----------------------
Arm Flapping: ${armFlapping}%
Spinning: ${spinning}%
Head Banging: ${headBanging}%

QUESTIONNAIRE DETAILS
---------------------
Social: ${social}%
Communication: ${communication}%
Behavioral: ${behavioral}%
Emotional: ${emotional}%

DETAILED EXPLANATION
--------------------
${aiExplanation}

GUIDANCE & NEXT STEPS
---------------------
${aiGuidance}

DISCLAIMER
----------
This is a risk assessment tool, NOT a clinical diagnosis. 
Please consult with a qualified healthcare professional for proper evaluation.
    `;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `autism-screening-report-${patientName.toLowerCase().replace(/\s+/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex-1 bg-gray-50 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-6 py-5 bg-white flex items-center justify-between z-10 shadow-sm relative">
        <h2 className="text-lg font-bold text-gray-900">Assessment Results</h2>
        <button
          onClick={handleDownloadPDF}
          disabled={loading}
          className={`flex items-center gap-2 px-4 py-2 rounded-full font-bold text-sm transition-all ${
            loading
              ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
              : 'bg-[#1A9E7A] text-white hover:bg-teal-700 shadow-md'
          }`}
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              Download Report
            </>
          )}
        </button>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto pb-24 md:pb-8 flex flex-col md:flex-row md:items-start md:p-8 md:gap-8">
        {/* Chart Section */}
        <div className="bg-white px-6 py-8 shadow-sm flex flex-col items-center border-b md:border md:rounded-3xl border-gray-100 md:w-[40%] flex-shrink-0">
          <div className="relative w-48 h-48 mb-2">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={70}
                  outerRadius={90}
                  startAngle={90}
                  endAngle={-270}
                  dataKey="value"
                  stroke="none"
                  cornerRadius={10}
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-4xl font-black text-gray-900">{fusionPercent}%</span>
              <span className="text-xs text-gray-500 font-semibold uppercase tracking-wider mt-1">Probability</span>
            </div>
          </div>

          <div className={`${riskBadge.bg} border px-4 py-1.5 rounded-full font-bold text-sm flex items-center gap-2 mb-2`}>
            <AlertTriangle className="w-4 h-4" />
            {riskBadge.text}
          </div>
          <p className="text-gray-500 text-sm font-medium">
            Video: <span className="text-gray-900 font-bold">{videoPercent}%</span> | 
            Questionnaire: <span className="text-gray-900 font-bold">{questionnairePercent}%</span>
          </p>
        </div>

        {/* Details Section */}
        <div className="md:flex-1 md:flex md:flex-col md:w-full">
          {/* AI Summary */}
          <div className="p-6 md:p-0 pb-2 md:pb-5">
            <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4 flex gap-3 text-amber-800">
              {loading ? (
                <Loader2 className="w-5 h-5 shrink-0 mt-0.5 animate-spin" />
              ) : (
                <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
              )}
              <p className="text-sm font-medium leading-relaxed">
                {loading ? 'Generating your personalized summary...' : aiSummary}
              </p>
            </div>
          </div>

          {/* Video Details */}
          <div className="px-6 md:px-0 py-2">
            <h3 className="font-bold text-gray-900 mb-3">Video Analysis</h3>
            <div className="bg-white rounded-2xl p-4 shadow-sm border border-gray-100 space-y-3">
              {[
                { label: 'Arm Flapping', value: armFlapping },
                { label: 'Spinning', value: spinning },
                { label: 'Head Banging', value: headBanging },
              ].map((item, i) => (
                <div key={i}>
                  <div className="flex justify-between text-sm mb-1 font-medium">
                    <span className="text-gray-700">{item.label}</span>
                    <span className="text-gray-900 font-bold">{item.value}%</span>
                  </div>
                  <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${item.value >= 70 ? 'bg-red-500' : item.value >= 40 ? 'bg-orange-500' : 'bg-green-500'}`}
                      style={{ width: `${item.value}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Questionnaire Details */}
          <div className="px-6 md:px-0 py-2">
            <h3 className="font-bold text-gray-900 mb-3">Questionnaire</h3>
            <div className="bg-white rounded-2xl p-4 shadow-sm border border-gray-100 space-y-3">
              {[
                { label: 'Social', value: social },
                { label: 'Communication', value: communication },
                { label: 'Behavioral', value: behavioral },
                { label: 'Emotional', value: emotional },
              ].map((item, i) => (
                <div key={i}>
                  <div className="flex justify-between text-sm mb-1 font-medium">
                    <span className="text-gray-700">{item.label}</span>
                    <span className="text-gray-900 font-bold">{item.value}%</span>
                  </div>
                  <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${item.value >= 70 ? 'bg-red-500' : item.value >= 40 ? 'bg-orange-500' : 'bg-green-500'}`}
                      style={{ width: `${item.value}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* AI Explanation */}
          <div className="px-6 md:px-0 py-4">
            <h3 className="font-bold text-gray-900 mb-3">Detailed Explanation</h3>
            <div className="bg-white rounded-2xl p-4 shadow-sm border border-gray-100">
              {loading ? (
                <div className="flex items-center gap-2 text-gray-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Generating explanation...</span>
                </div>
              ) : (
                <p className="text-sm text-gray-700 leading-relaxed">{aiExplanation}</p>
              )}
            </div>
          </div>

          {/* AI Guidance */}
          <div className="px-6 md:px-0 py-2">
            <h3 className="font-bold text-gray-900 mb-3">Guidance & Next Steps</h3>
            <div className="bg-teal-50 border border-teal-200 rounded-2xl p-4">
              {loading ? (
                <div className="flex items-center gap-2 text-teal-600">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Generating guidance...</span>
                </div>
              ) : (
                <p className="text-sm text-teal-800 leading-relaxed">{aiGuidance}</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
