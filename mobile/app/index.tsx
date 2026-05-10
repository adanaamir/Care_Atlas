import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator, ScrollView, Platform } from 'react-native';
import { useRouter } from 'expo-router';
import { useState } from 'react';
// Remove SafeAreaView from here to avoid name conflict if needed, or just use RN one
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { AlertCircle, Stethoscope, Baby, Activity, Heart, Briefcase } from 'lucide-react-native';

const NEEDS = [
  { id: 'general', label: 'General Care', icon: Stethoscope },
  { id: 'maternity', label: 'Maternity', icon: Baby },
  { id: 'surgery', label: 'Surgery', icon: Briefcase },
  { id: 'icu', label: 'ICU', icon: Activity },
  { id: 'dialysis', label: 'Dialysis', icon: Heart },
];

export default function HomeScreen() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const insets = useSafeAreaInsets();

  const handleEmergency = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      router.push({ pathname: '/results', params: { need: 'emergency' } });
    }, 500);
  };

  const handleSpecificNeed = (needId: string) => {
    router.push({ pathname: '/results', params: { need: needId } });
  };

  const handleVoice = () => {
    if (Platform.OS !== 'web') {
      alert("Voice search is currently available in the Web Demo. Try saying 'Take me to the nearest ICU'");
      return;
    }

    // Web Speech API
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech recognition not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    alert("Listening... Say a medical need (e.g., ICU, Maternity, Surgery)");

    recognition.onresult = (event: any) => {
      const speech = event.results[0][0].transcript.toLowerCase();
      console.log("Speech detected:", speech);

      // Simple intent detection
      if (speech.includes('icu')) handleSpecificNeed('icu');
      else if (speech.includes('maternity') || speech.includes('baby') || speech.includes('birth')) handleSpecificNeed('maternity');
      else if (speech.includes('surgery') || speech.includes('operation')) handleSpecificNeed('surgery');
      else if (speech.includes('dialysis') || speech.includes('kidney')) handleSpecificNeed('dialysis');
      else if (speech.includes('emergency') || speech.includes('help')) handleSpecificNeed('emergency');
      else {
        alert(`You said: "${speech}". Try saying "ICU" or "Maternity"`);
      }
    };

    recognition.onerror = (event: any) => {
      console.error("Speech recognition error", event.error);
      alert("Could not hear you clearly. Please try again.");
    };

    recognition.start();
  };


  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.maxWidthContainer}>
          <View style={styles.header}>
            <Text style={styles.title}>CareAtlas 🇳🇬</Text>
            <Text style={styles.subtitle}>Instant Healthcare Routing</Text>
          </View>

          <View style={styles.mainContent}>

            <TouchableOpacity 
              style={styles.emergencyButton} 
              onPress={handleEmergency}
              disabled={loading}
              activeOpacity={0.8}
            >
              {loading ? (
                <ActivityIndicator size="large" color="#ffffff" />
              ) : (
                <>
                  <AlertCircle size={80} color="#ffffff" strokeWidth={2.5} />
                  <Text style={styles.emergencyText}>FIND EMERGENCY CARE</Text>
                  <Text style={styles.emergencySubtext}>Tap for nearest capable hospital</Text>
                </>
              )}
            </TouchableOpacity>

            <TouchableOpacity 
              style={styles.voiceButton}
              onPress={handleVoice}
            >
              <Activity size={20} color="#dc3545" />
              <Text style={styles.voiceText}>Voice Search</Text>
            </TouchableOpacity>

          </View>

          <View style={styles.secondaryContent}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Other Medical Needs</Text>
              <TouchableOpacity onPress={() => router.push('/deserts')}>
                <Text style={styles.desertLink}>Desert Map</Text>
              </TouchableOpacity>
            </View>
            <View style={styles.grid}>
              {NEEDS.map((need) => (
                <TouchableOpacity 
                  key={need.id} 
                  style={styles.needCard}
                  onPress={() => handleSpecificNeed(need.id)}
                  activeOpacity={0.7}
                >
                  <need.icon size={32} color="#1a7f37" strokeWidth={2} />
                  <Text style={styles.needText}>{need.label}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  scrollContent: {
    flexGrow: 1,
    alignItems: 'center',
    paddingBottom: 40,
  },
  maxWidthContainer: {
    width: '100%',
    maxWidth: 600, // Keeps it from stretching too far on web
  },
  header: {
    padding: 24,
    alignItems: 'center',
  },
  title: {
    fontSize: 32,
    fontWeight: '900',
    color: '#1a7f37',
    marginBottom: 2,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 16,
    color: '#6c757d',
    fontWeight: '600',
  },
  mainContent: {
    alignItems: 'center',
    paddingVertical: 20,
  },

  emergencyButton: {
    width: 280,
    height: 280,
    backgroundColor: '#dc3545',
    borderRadius: 140,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#dc3545',
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.4,
    shadowRadius: 16,
    elevation: 20,
    borderWidth: 8,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  emergencyText: {
    color: '#ffffff',
    fontSize: 24,
    fontWeight: '900',
    marginTop: 20,
    textAlign: 'center',
    paddingHorizontal: 20,
  },
  emergencySubtext: {
    color: 'rgba(255,255,255,0.9)',
    fontSize: 14,
    marginTop: 8,
    fontWeight: '600',
  },
  voiceButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 32,
    backgroundColor: '#fff',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 30,
    borderWidth: 1.5,
    borderColor: '#dc3545',
    gap: 10,
    shadowColor: '#dc3545',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  voiceText: {
    color: '#dc3545',
    fontWeight: '800',
    fontSize: 15,
  },
  secondaryContent: {
    padding: 24,
    backgroundColor: '#ffffff',
    borderRadius: 32,
    marginHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.05,
    shadowRadius: 20,
    elevation: 5,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
    paddingHorizontal: 4,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '800',
    color: '#212529',
  },
  desertLink: {
    color: '#1a7f37',
    fontWeight: '800',
    fontSize: 15,
    textDecorationLine: 'underline',
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  needCard: {
    flexBasis: '48%', // 2 columns on mobile, will wrap properly
    flexGrow: 1,
    backgroundColor: '#f8f9fa',
    paddingVertical: 20,
    paddingHorizontal: 12,
    borderRadius: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#f1f3f5',
  },
  needText: {
    marginTop: 10,
    fontSize: 14,
    fontWeight: '700',
    color: '#495057',
    textAlign: 'center',
  },
});

