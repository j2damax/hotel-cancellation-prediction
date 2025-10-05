import type { FeatureImportance } from '../types/api';

interface FeatureContributionsProps {
  features: FeatureImportance[];
  isLocal?: boolean;
}

export function FeatureContributions({ 
  features, 
  isLocal = false 
}: FeatureContributionsProps) {
  if (!features || features.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">
          {isLocal ? 'Local' : 'Top'} Feature Contributions
        </h3>
        <p className="text-sm text-gray-500">
          {isLocal 
            ? 'Local SHAP not available – showing top global features'
            : 'Feature importance data not available'}
        </p>
      </div>
    );
  }

  const topFeatures = features.slice(0, 3);

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h3 className="text-lg font-semibold mb-4">
        Top 3 Feature Contributions
      </h3>
      
      {!isLocal && (
        <p className="text-xs text-gray-500 mb-3">
          ℹ️ Showing global feature importance
        </p>
      )}

      <div className="space-y-3">
        {topFeatures.map((feature, index) => (
          <div key={feature.feature} className="flex items-center gap-3">
            <div className="flex-shrink-0 w-6 h-6 flex items-center justify-center bg-blue-100 text-blue-700 rounded-full text-xs font-semibold">
              {index + 1}
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium text-gray-900">
                {feature.human_readable || feature.feature}
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                <div
                  className="bg-blue-600 h-2 rounded-full"
                  style={{
                    width: `${Math.min(100, (feature.mean_abs_shap / (topFeatures[0]?.mean_abs_shap || 1)) * 100)}%`,
                  }}
                />
              </div>
            </div>
            <div className="flex-shrink-0 text-xs text-gray-500">
              {feature.mean_abs_shap.toFixed(3)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
