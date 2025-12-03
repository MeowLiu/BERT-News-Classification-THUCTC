import { PredictResponse, ModelType } from '../types';
import { NEWS_CATEGORIES } from '../constants';

export const mockAnalyze = (text: string, model: ModelType): Promise<PredictResponse> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Simple heuristic simulation based on keywords or random if plain
      let topCategory = "其他";

      if (text.includes("赛") || text.includes("赢") || text.includes("球") || text.includes("冠")) {
        topCategory = "体育";
      } else if (text.includes("股") || text.includes("金") || text.includes("市") || text.includes("资")) {
        topCategory = "财经";
      } else if (text.includes("芯") || text.includes("网") || text.includes("AI") || text.includes("码")) {
        topCategory = "科技";
      } else {
        topCategory = NEWS_CATEGORIES[Math.floor(Math.random() * NEWS_CATEGORIES.length)];
      }

      // Generate random probabilities
      const remainingCategories = NEWS_CATEGORIES.filter(c => c !== topCategory);
      // Shuffle remaining
      remainingCategories.sort(() => 0.5 - Math.random());

      const topScore = 0.7 + (Math.random() * 0.25); // 0.70 - 0.95
      let remainingScore = 1 - topScore;

      const probs = [{ name: topCategory, value: parseFloat((topScore * 100).toFixed(1)) }];

      // Fill top 4 others
      for (let i = 0; i < 4; i++) {
        const val = Math.random() * remainingScore;
        remainingScore -= val;
        probs.push({ name: remainingCategories[i], value: parseFloat((val * 100).toFixed(1)) });
      }

      // Sort by value desc
      probs.sort((a, b) => b.value - a.value);

      resolve({
        category: topCategory,
        confidence: topScore,
        probabilities: probs,
        inference_time_ms: Math.floor(Math.random() * 200) + (model === ModelType.BERT ? 600 : 750) // RoBERTa slightly slower usually
      });
    }, 800);
  });
};