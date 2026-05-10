import { View, Text, StyleSheet, FlatList, ActivityIndicator, TouchableOpacity, Alert } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { getCurrentLocation } from '../lib/location';
import { findNearestFacilities, NearestFacilityResult, NeedType } from '../lib/api';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MapPin, Navigation, Star, Clock, AlertTriangle, ShieldCheck } from 'lucide-react-native';

export default function ResultsScreen() {
  const { need } = useLocalSearchParams<{ need: string }>();
  const router = useRouter();
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<NearestFacilityResult[]>([]);
  const [fallbackMessage, setFallbackMessage] = useState<string | null>(null);
  const [isDemoMode, setIsDemoMode] = useState(false);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        setError(null);
        setIsDemoMode(false);
        
        // 1. Get Location
        const location = await getCurrentLocation();
        
        // 2. Fetch nearest facilities
        let response;
        try {
          response = await findNearestFacilities(location.lat, location.lon, need as NeedType);
          if (response.results.length === 0) {
            throw new Error("No results found near your current location.");
          }
        } catch (err) {
          console.log("Outside Nigeria or location failed, falling back to Lagos demo coords", err);
          // Lagos fallback coordinates
          response = await findNearestFacilities(6.5244, 3.3792, need as NeedType);
          setIsDemoMode(true);
        }

        setResults(response.results);

        if (response.fallback_used) {
          setFallbackMessage(response.message);
        }
      } catch (err: any) {
        setError(err.message || 'An unexpected error occurred');
        Alert.alert("Error", "Could not fetch facilities. Please try again.");
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [need]);

  const renderTrustBadge = (score: number, grade: string) => {
    let color = '#1a7f37'; // A
    if (grade === 'B' || grade === 'C') color = '#fd7e14';
    if (grade === 'D' || grade === 'F') color = '#dc3545';

    return (
      <View style={[styles.trustBadge, { backgroundColor: color + '15', borderColor: color }]}>
        <ShieldCheck size={14} color={color} />
        <Text style={[styles.trustText, { color }]}>
          Trust: {grade} ({(score * 100).toFixed(0)}%)
        </Text>
      </View>
    );
  };

  const renderFacility = ({ item, index }: { item: NearestFacilityResult; index: number }) => (
    <TouchableOpacity 
      style={styles.card}
      onPress={() => router.push({ pathname: `/facility/${item.facility_id}`, params: { data: JSON.stringify(item) } })}
      activeOpacity={0.7}
    >
      <View style={styles.cardHeader}>
        <View style={styles.rankBadge}>
          <Text style={styles.rankText}>#{index + 1}</Text>
        </View>
        <Text style={styles.facilityName} numberOfLines={2}>{item.facility_name}</Text>
      </View>

      <View style={styles.cardBody}>
        <Text style={styles.categoryText}>
          {item.category || item.facility_type || 'Healthcare Facility'} • {item.functional_status || 'Unknown Status'}
        </Text>

        <View style={styles.statsRow}>
          <View style={styles.stat}>
            <MapPin size={16} color="#6c757d" />
            <Text style={styles.statText}>{item.distance_km.toFixed(1)} km</Text>
          </View>
          <View style={styles.stat}>
            <Clock size={16} color="#6c757d" />
            <Text style={styles.statText}>~{item.estimated_travel_min} min</Text>
          </View>
        </View>

        <View style={styles.badgesRow}>
          {renderTrustBadge(item.trust_score, item.trust_grade)}
          {item.matched_capabilities.length > 0 && (
            <View style={styles.capBadge}>
              <Text style={styles.capText}>{item.matched_capabilities.length} capabilities match</Text>
            </View>
          )}
        </View>
      </View>
      
      <View style={styles.cardFooter}>
        <Text style={styles.actionText}>View Details</Text>
        <Navigation size={16} color="#0969da" />
      </View>
    </TouchableOpacity>
  );

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#1a7f37" />
        <Text style={styles.loadingText}>Locating nearest facilities...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View style={styles.maxWidthContainer}>
          <Text style={styles.headerTitle}>
            Nearest for <Text style={{fontWeight: '900'}}>{(need as string).toUpperCase()}</Text>
          </Text>
        </View>
      </View>
      
      <View style={styles.mainContent}>
        <View style={styles.maxWidthContainer}>
          {isDemoMode && (
            <View style={styles.demoBanner}>
              <MapPin size={16} color="#fff" />
              <Text style={styles.demoText}>📍 Viewing facilities near Lagos (Demo Mode)</Text>
            </View>
          )}

          {fallbackMessage && (
            <View style={styles.warningBox}>
              <AlertTriangle size={20} color="#856404" />
              <Text style={styles.warningText}>{fallbackMessage}</Text>
            </View>
          )}


          {error ? (
            <View style={styles.centered}>
              <Text style={styles.errorText}>{error}</Text>
              <TouchableOpacity style={styles.retryBtn} onPress={() => router.back()}>
                <Text style={styles.retryText}>Go Back</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <FlatList
              data={results}
              keyExtractor={(item) => item.facility_id}
              renderItem={renderFacility}
              contentContainerStyle={styles.listContainer}
              showsVerticalScrollIndicator={false}
            />
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f1f3f5',
  },
  mainContent: {
    flex: 1,
    alignItems: 'center',
  },
  maxWidthContainer: {
    width: '100%',
    maxWidth: 600,
    flex: 1,
  },
  demoBanner: {
    flexDirection: 'row',
    backgroundColor: '#0969da',
    padding: 12,
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 8,
    alignItems: 'center',
    gap: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  demoText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '700',
  },



  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#495057',
    fontWeight: '500',
  },
  header: {
    backgroundColor: '#1a7f37',
    padding: 16,
    paddingTop: 8,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
  listContainer: {
    padding: 16,
    gap: 16,
    width: '100%',
  },

  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  cardHeader: {
    flexDirection: 'row',
    padding: 16,
    paddingBottom: 8,
    alignItems: 'flex-start',
    gap: 12,
  },
  rankBadge: {
    backgroundColor: '#e9ecef',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  rankText: {
    fontWeight: '800',
    color: '#495057',
    fontSize: 14,
  },
  facilityName: {
    flex: 1,
    fontSize: 18,
    fontWeight: '700',
    color: '#212529',
    lineHeight: 24,
  },
  cardBody: {
    paddingHorizontal: 16,
    paddingBottom: 16,
  },
  categoryText: {
    fontSize: 14,
    color: '#6c757d',
    marginBottom: 12,
  },
  statsRow: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 16,
  },
  stat: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  statText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#495057',
  },
  badgesRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  trustBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
  },
  trustText: {
    fontSize: 12,
    fontWeight: '700',
  },
  capBadge: {
    backgroundColor: '#e7f5ff',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#74c0fc',
  },
  capText: {
    color: '#1864ab',
    fontSize: 12,
    fontWeight: '600',
  },
  cardFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 12,
    backgroundColor: '#f8f9fa',
    borderTopWidth: 1,
    borderTopColor: '#e9ecef',
  },
  actionText: {
    color: '#0969da',
    fontWeight: '700',
    fontSize: 14,
  },
  warningBox: {
    flexDirection: 'row',
    backgroundColor: '#fff3cd',
    padding: 12,
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 8,
    alignItems: 'center',
    gap: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#ffc107',
  },
  warningText: {
    flex: 1,
    color: '#856404',
    fontSize: 14,
  },
  errorText: {
    color: '#dc3545',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 16,
  },
  retryBtn: {
    backgroundColor: '#212529',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryText: {
    color: '#fff',
    fontWeight: '600',
  },
});
