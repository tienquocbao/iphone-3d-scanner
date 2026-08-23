import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "viewfinder")
                .font(.system(size: 64))

            Text("iPhone 3D Scanner")
                .font(.title)
                .bold()

            Text("Phase 0B")
                .font(.headline)

            Text("Native iOS build pipeline ready")
                .foregroundStyle(.secondary)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
