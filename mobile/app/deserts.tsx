import { View, Text, StyleSheet, FlatList, ActivityIndicator, TouchableOpacity, ScrollView } from 'react-native';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { API_BASE_URL } from '../lib/api';
import { SafeAreaView } from 'react-native-safe-area-context';
import { AlertTriangle, Map, BarChart2, Globe, ChevronRight } from 'lucide-react-native';

interface DesertRegion {
  pin_code: string;
  state: string;
  facility_count: number;
  desert_risk_score: number;
  is_high_risk: boolean;
  desert_categories: string[];
  coverage: {
    icu: number;
    dialysis: number;
    emergency: number;
    surgery: number;
  };
}

export default function DesertsScreen() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<{ regions: DesertRegion[], summary: any } | null>(null);

  useEffect(() => {
    async function fetchDeserts() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/deserts?limit=20&high_risk_only=true`);
        const json = await response.json();
        setData(json);
      } catch (err) {
        console.error("Failed to fetch medical deserts", err);
      } finally {
        setLoading(false);
      }
    }
    fetchDeserts();
  }, []);

  const renderRegion = ({ item }: { item: DesertRegion }) => (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={[styles.riskBadge, { backgroundColor: item.is_high_risk ? '#fff5f5' : '#f8f9fa' }]}>
          <Text style={[styles.riskText, { color: item.is_high_risk ? '#e03131' : '#495057' }]}>
            Score: {(item.desert_risk_score * 100).toFixed(0)}
          </Text>
        </View>
        <Text style={styles.regionTitle}>{item.pin_code} - {item.state}</Text>
      </View>

      <View style={styles.statsGrid}>
        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Facilities</Text>
          <Text style={styles.statValue}>{item.facility_count}</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={styles.statLabel}>ICU Coverage</Text>
          <Text style={[styles.statValue, { color: item.coverage.icu < 0.1 ? '#e03131' : '#2f9e44' }]}>
            {(item.coverage.icu * 100).toFixed(0)}%
          </Text>
        </View>
        <View style={styles.statItem}>
          <Text style={styles.statLabel}>Emergency</Text>
          <Text style={[styles.statValue, { color: item.coverage.emergency < 0.1 ? '#e03131' : '#2f9e44' }]}>
            {(item.coverage.emergency * 100).toFixed(0)}%
          </Text>
        </View>
      </View>

      <View style={styles.tagsContainer}>
        {item.desert_categories.map((cat, idx) => (
          <View key={idx} style={styles.tag}>
            <Text style={styles.tagText}>{cat.replace('_DESERT', '')}</Text>
          </View>
        ))}
      </View>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#1a7f37" />
        <Text style={styles.loadingText}>Analyzing healthcare infrastructure...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.maxWidthContainer}>
          <View style={styles.summaryCard}>
            <View style={styles.summaryHeader}>
              <AlertTriangle size={24} color="#f08c00" />
              <Text style={styles.summaryTitle}>National Healthcare Gaps</Text>
            </View>
            <View style={styles.summaryStats}>
              <View style={styles.summaryStat}>
                <Text style={styles.summaryStatVal}>{data?.summary.high_risk_regions}</Text>
                <Text style={styles.summaryStatLab}>Critical LGA Deserts</Text>
              </View>
              <View style={styles.summaryStat}>
                <Text style={styles.summaryStatVal}>{data?.summary.high_risk_percentage}%</Text>
                <Text style={styles.summaryStatLab}>High Risk Areas</Text>
              </View>
            </View>
            <Text style={styles.summaryDesc}>
              These regions represent areas where essential medical services are significantly below national accessibility benchmarks.
            </Text>
          </View>

          <Text style={styles.sectionTitle}>High-Risk Regions (Top 20)</Text>
          
          {data?.regions.map((region, idx) => (
            <View key={region.pin_code + idx}>
              {renderRegion({ item: region })}
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f1f3f5',
  },
  scrollContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 16,
    paddingBottom: 40,
  },
  maxWidthContainer: {
    width: '100%',
    maxWidth: 600,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    color: '#495057',
  },
  summaryCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
    borderLeftWidth: 6,
    borderLeftColor: '#f08c00',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  summaryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 16,
  },
  summaryTitle: {
    fontSize: 20,
    fontWeight: '800',
    color: '#212529',
  },
  summaryStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  summaryStat: {
    alignItems: 'center',
    flex: 1,
  },
  summaryStatVal: {
    fontSize: 24,
    fontWeight: '900',
    color: '#f08c00',
  },
  summaryStatLab: {
    fontSize: 12,
    color: '#868e96',
    marginTop: 4,
  },
  summaryDesc: {
    fontSize: 14,
    color: '#495057',
    lineHeight: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#343a40',
    marginBottom: 16,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  riskText: {
    fontSize: 12,
    fontWeight: '700',
  },
  regionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#212529',
    flex: 1,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  statItem: {
    flex: 1,
  },
  statLabel: {
    fontSize: 10,
    color: '#868e96',
    marginBottom: 2,
    textTransform: 'uppercase',
  },
  statValue: {
    fontSize: 14,
    fontWeight: '700',
    color: '#495057',
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  tag: {
    backgroundColor: '#f1f3f5',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  tagText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#495057',
    textTransform: 'uppercase',
  },
});
