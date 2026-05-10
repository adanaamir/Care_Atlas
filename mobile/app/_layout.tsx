import { Stack } from 'expo-router';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <StatusBar style="auto" />
      <Stack
        screenOptions={{
          headerStyle: {
            backgroundColor: '#1a7f37',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="results" options={{ title: 'Nearest Facilities' }} />
        <Stack.Screen name="facility/[id]" options={{ title: 'Facility Details' }} />
        <Stack.Screen name="deserts" options={{ title: 'Medical Desert Intelligence' }} />
      </Stack>

    </SafeAreaProvider>
  );
}
