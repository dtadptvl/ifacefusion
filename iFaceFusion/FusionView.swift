import SwiftUI
import PhotosUI

struct FusionView: View {
    @EnvironmentObject var store: FusionStore
    @State private var sourceItem: PhotosPickerItem?
    @State private var targetItem: PhotosPickerItem?
    @State private var sharing = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    HStack(spacing: 12) {
                        PhotoCard(title: "Source", image: store.source, item: $sourceItem)
                        PhotoCard(title: "Target", image: store.target, item: $targetItem)
                    }

                    processorSection

                    if store.showAdvanced {
                        AdvancedView(settings: $store.settings, selected: store.selected)
                    }

                    Button(action: store.process) {
                        HStack {
                            if store.isProcessing {
                                ProgressView().tint(.white)
                            }
                            Text(store.isProcessing ? store.status : "Process")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity, minHeight: 52)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!store.canProcess)
                    .accessibilityHint("Runs all enabled processors in order")

                    if let result = store.result {
                        resultSection(result)
                    }
                }
                .padding(16)
            }
            .navigationTitle("iFaceFusion")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(store.showAdvanced ? "Done" : "Advanced") {
                        withAnimation {
                            store.showAdvanced.toggle()
                        }
                    }
                }
            }
            .task(id: sourceItem) {
                await store.load(sourceItem, asSource: true)
            }
            .task(id: targetItem) {
                await store.load(targetItem, asSource: false)
            }
            .alert(
                "iFaceFusion",
                isPresented: Binding(
                    get: { store.errorMessage != nil },
                    set: { if !$0 { store.errorMessage = nil } }
                )
            ) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(store.errorMessage ?? "")
            }
            .sheet(isPresented: $sharing) {
                if let result = store.result {
                    ShareSheet(items: [result])
                }
            }
        }
    }

    private var processorSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Processors").font(.title2.bold())
                Spacer()
                Text("\(store.selected.count) enabled")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            ForEach(ProcessorID.allCases) { processor in
                Button {
                    if store.selected.contains(processor) {
                        store.selected.remove(processor)
                    } else {
                        store.selected.insert(processor)
                    }
                } label: {
                    HStack(spacing: 14) {
                        Image(systemName: processor.systemImage)
                            .frame(width: 28)
                            .font(.title3)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(processor.title).fontWeight(.semibold)
                            Text(processor.subtitle)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName: store.selected.contains(processor) ? "checkmark.circle.fill" : "circle")
                            .font(.title2)
                    }
                    .contentShape(Rectangle())
                    .frame(minHeight: 52)
                }
                .buttonStyle(.plain)
                .accessibilityValue(store.selected.contains(processor) ? "Enabled" : "Disabled")
            }
        }
        .padding(16)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }

    private func resultSection(_ image: UIImage) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Before / After").font(.title2.bold())
            if let target = store.target {
                BeforeAfterView(before: target, after: image)
            }
            HStack {
                Button("Save to Photos", systemImage: "square.and.arrow.down", action: store.save)
                    .buttonStyle(.borderedProminent)
                Button("Share", systemImage: "square.and.arrow.up") {
                    sharing = true
                }
                .buttonStyle(.bordered)
            }
        }
    }
}

private struct PhotoCard: View {
    let title: String
    let image: UIImage?
    @Binding var item: PhotosPickerItem?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            PhotosPicker(selection: $item, matching: .images) {
                ZStack {
                    RoundedRectangle(cornerRadius: 18)
                        .fill(.secondary.opacity(0.12))
                    if let image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "photo.badge.plus").font(.title)
                            Text("Choose photo").font(.caption)
                        }
                    }
                }
                .frame(height: 180)
                .clipShape(RoundedRectangle(cornerRadius: 18))
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Choose \(title) image")
        }
        .frame(maxWidth: .infinity)
    }
}

private struct BeforeAfterView: View {
    let before: UIImage
    let after: UIImage
    @State private var afterVisible = true

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            Image(uiImage: afterVisible ? after : before)
                .resizable()
                .scaledToFit()
                .clipShape(RoundedRectangle(cornerRadius: 18))
            Button(afterVisible ? "After" : "Before") {
                afterVisible.toggle()
            }
            .buttonStyle(.borderedProminent)
            .padding(12)
        }
    }
}

private struct AdvancedView: View {
    @Binding var settings: ProcessorSettings
    let selected: Set<ProcessorID>

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Advanced").font(.title2.bold())
            if selected.contains(.faceEnhance) {
                LabeledContent("Face enhance") {
                    Slider(value: $settings.faceEnhanceBlend, in: 0...1)
                }
            }
            if selected.contains(.age) {
                LabeledContent("Age change") {
                    Slider(value: $settings.ageDirection, in: -100...100, step: 1)
                }
            }
            if selected.contains(.expressionRestore) {
                LabeledContent("Expression restore") {
                    Slider(value: $settings.expressionFactor, in: 0...100, step: 1)
                }
            }
            if selected.contains(.faceEdit) {
                LabeledContent("Smile") {
                    Slider(value: $settings.faceEditSmile, in: -1...1)
                }
            }
            if selected.contains(.colourise) {
                LabeledContent("Colour blend") {
                    Slider(value: $settings.colourBlend, in: 0...1)
                }
            }
            if selected.contains(.frameEnhance) {
                LabeledContent("Enhance blend") {
                    Slider(value: $settings.frameEnhanceBlend, in: 0...1)
                }
            }
            Text("Models download only when first required and remain cached for offline use. Large models can require substantial storage.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
    }
}