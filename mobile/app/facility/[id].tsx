import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Linking, Platform } from 'react-native';
import { useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { NearestFacilityResult } from '../../lib/api';
import { Phone, Map, MapPin, Clock, ShieldCheck, Info, CheckCircle2, AlertCircle } from 'lucide-react-native';

export default function FacilityScreen() {
  const { data } = useLocalSearchParams<{ data: string }>();
  
  let facility: NearestFacilityResult | null = null;
  try {
    facility = data ? JSON.parse(data) : null;
  } catch (e) {
    console.error("Failed to parse facility data", e);
  }

  if (!facility) {
    return (
      <View style={styles.centered}>
        <Text>Facility not found.</Text>
      </View>
    );
  }

  const { facility_meta } = facility;

  const openMaps = () => {
    // Attempt to get coordinates from meta
    const lat = facility.facility_meta?.latitude || facility.facility_meta?.lat;
    const lng = facility.facility_meta?.longitude || facility.facility_meta?.lon || facility.facility_meta?.lng;
    
    if (!lat || !lng) {
      alert("Coordinates not available for this facility.");
      return;
    }

    const label = encodeURIComponent(facility.facility_name);
    
    if (Platform.OS === 'web') {
      // Use standard Google Maps URL for browsers
      Linking.openURL(`https://www.google.com/maps/search/?api=1&query=${lat},${lng}`);
    } else if (Platform.OS === 'ios') {
      Linking.openURL(`maps:0,0?q=${label}&ll=${lat},${lng}`);
    } else {
      Linking.openURL(`geo:0,0?q=${lat},${lng}(${label})`);
    }
  };

  const handleCall = () => {
    const phone = facility.facility_meta?.phone || "0800-CARE-ATLAS";
    Linking.openURL(`tel:${phone}`);
  };

  const getTrustColor = (grade: string) => {
    if (grade === 'A') return '#1a7f37';
    if (grade === 'B' || grade === 'C') return '#fd7e14';
    return '#dc3545';
  };

  const trustColor = getTrustColor(facility.trust_grade);

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ paddingBottom: 40 }}>
      {/* Header Info */}
      <View style={styles.header}>
        <Text style={styles.title}>{facility.facility_name}</Text>
        <Text style={styles.subtitle}>
          {facility.category || facility.facility_type || 'Healthcare Facility'}
        </Text>
        
        <View style={styles.locationRow}>
          <MapPin size={16} color="#6c757d" />
          <Text style={styles.locationText}>
            {facility_meta.lga || facility_meta.city}, {facility_meta.state}
          </Text>
        </View>

        {facility.functional_status && (
          <View style={styles.statusRow}>
            {facility.functional_status.toLowerCase().includes('functional') ? (
              <CheckCircle2 size={16} color="#1a7f37" />
            ) : (
              <AlertCircle size={16} color="#dc3545" />
            )}
            <Text style={[styles.statusText, { color: facility.functional_status.toLowerCase().includes('functional') ? '#1a7f37' : '#dc3545' }]}>
              {facility.functional_status}
            </Text>
          </View>
        )}
      </View>

      {/* Primary Actions */}
      <View style={styles.actionGrid}>
        <TouchableOpacity style={styles.actionBtn} onPress={openMaps} activeOpacity={0.8}>
          <View style={[styles.actionIcon, { backgroundColor: '#e7f5ff' }]}>
            <Navigation size={24} color="#1864ab" />
          </View>
          <Text style={styles.actionLabel}>Directions</Text>
          <Text style={styles.actionSub}>{facility.distance_km.toFixed(1)} km (~{facility.estimated_travel_min}m)</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.actionBtn} onPress={handleCall} activeOpacity={0.8}>
          <View style={[styles.actionIcon, { backgroundColor: '#ebfbee' }]}>
            <Phone size={24} color="#2b8a3e" />
          </View>
          <Text style={styles.actionLabel}>Call</Text>
          <Text style={styles.actionSub}>Helpline</Text>
        </TouchableOpacity>

      </View>

      {/* Trust Score Panel */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Trust Evaluation</Text>
        <View style={[styles.trustPanel, { borderColor: trustColor + '40', backgroundColor: trustColor + '08' }]}>
          <View style={styles.trustHeader}>
            <ShieldCheck size={32} color={trustColor} />
            <View style={styles.trustScoreBlock}>
              <Text style={[styles.trustGrade, { color: trustColor }]}>{facility.trust_grade}</Text>
              <Text style={styles.trustPct}>{(facility.trust_score * 100).toFixed(0)}% Trust Score</Text>
            </View>
          </View>
          <Text style={styles.trustDesc}>
            Based on data consistency, facility level, and functional status reported in the National Health Facility Registry.
          </Text>
        </View>
      </View>

      {/* Capabilities */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Confirmed Capabilities</Text>
        <View style={styles.capsContainer}>
          {facility.matched_capabilities.length > 0 ? (
            facility.matched_capabilities.map((cap, i) => (
              <View key={i} style={styles.capPill}>
                <CheckCircle2 size={14} color="#0969da" />
                <Text style={styles.capText}>{cap}</Text>
              </View>
            ))
          ) : (
            <Text style={styles.emptyText}>Specific capabilities not detailed. Likely provides standard {facility.category?.toLowerCase() || 'primary'} services.</Text>
          )}
        </View>
      </View>

      {/* Details */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Facility Details</Text>
        <View style={styles.detailCard}>
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Operator Type</Text>
            <Text style={styles.detailValue}>{facility_meta.operator_type || 'Unknown'}</Text>
          </View>
          <View style={styles.divider} />
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Facility ID</Text>
            <Text style={styles.detailValue} numberOfLines={1} ellipsizeMode="middle">{facility.facility_id}</Text>
          </View>
          {facility_meta.accessibility_note ? (
            <>
              <View style={styles.divider} />
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Accessibility</Text>
                <Text style={styles.detailValue}>{facility_meta.accessibility_note}</Text>
              </View>
            </>
          ) : null}
        </View>
      </View>

    </ScrollView>
  );
}

// Need to import Navigation from lucide-react-native
import { Navigation } from 'lucide-react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    backgroundColor: '#fff',
    padding: 24,
    paddingTop: 32,
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
  },
  title: {
    fontSize: 24,
    fontWeight: '800',
    color: '#212529',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    color: '#495057',
    fontWeight: '600',
    marginBottom: 12,
  },
  locationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
  },
  locationText: {
    fontSize: 15,
    color: '#6c757d',
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 4,
  },
  statusText: {
    fontSize: 14,
    fontWeight: '600',
  },
  actionGrid: {
    flexDirection: 'row',
    padding: 16,
    gap: 16,
  },
  actionBtn: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  actionIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  actionLabel: {
    fontSize: 16,
    fontWeight: '700',
    color: '#212529',
  },
  actionSub: {
    fontSize: 13,
    color: '#6c757d',
    marginTop: 4,
  },
  section: {
    padding: 24,
    paddingTop: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#343a40',
    marginBottom: 16,
  },
  trustPanel: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
  },
  trustHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: 12,
  },
  trustScoreBlock: {
    flex: 1,
  },
  trustGrade: {
    fontSize: 28,
    fontWeight: '900',
  },
  trustPct: {
    fontSize: 14,
    fontWeight: '600',
    color: '#495057',
  },
  trustDesc: {
    fontSize: 14,
    color: '#6c757d',
    lineHeight: 20,
  },
  capsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  capPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f1f3f5',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    gap: 6,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  capText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#212529',
  },
  emptyText: {
    color: '#6c757d',
    fontStyle: 'italic',
    lineHeight: 20,
  },
  detailCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
  },
  detailLabel: {
    color: '#6c757d',
    fontSize: 14,
    fontWeight: '500',
  },
  detailValue: {
    color: '#212529',
    fontSize: 14,
    fontWeight: '600',
    maxWidth: '60%',
    textAlign: 'right',
  },
  divider: {
    height: 1,
    backgroundColor: '#e9ecef',
    marginVertical: 4,
  },
});
