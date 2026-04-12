import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components"
import "views"

ApplicationWindow {
    id: root
    width: 1680
    height: 980
    minimumWidth: 1180
    minimumHeight: 720
    visible: true
    title: "Crypto Strategy Lab"
    color: "#070A0F"

    property int navIndex: 0

    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#090D14" }
            GradientStop { position: 1.0; color: "#06080D" }
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        NavigationRail {
            id: nav
            Layout.fillHeight: true
            currentIndex: root.navIndex
            onIndexChanged: function(i) { root.navIndex = i }
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10

            TopBar {
                Layout.fillWidth: true
                onStartClicked: appState.startResearch()
                onPauseClicked: appState.pauseResearch()
                onStopClicked: appState.stopResearch()
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 10

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 14
                    color: "#0E1420"
                    border.color: "#1A2638"

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 8
                        spacing: 8

                        TabBar {
                            id: tabBar
                            Layout.fillWidth: true
                            background: Rectangle { color: "#0D1523"; radius: 10 }
                            TabButton { text: "Overview" }
                            TabButton { text: "Strategies" }
                            TabButton { text: "Evolution" }
                            TabButton { text: "Neural" }
                            TabButton { text: "Results" }
                        }

                        StackLayout {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            currentIndex: tabBar.currentIndex
                            OverviewView {}
                            StrategyView {}
                            EvolutionView {}
                            NeuralView {}
                            ResultsView {}
                        }
                    }
                }

                InspectorPanel {
                    id: inspector
                    Layout.preferredWidth: 360
                    Layout.fillHeight: true
                    strategy: appState.selectedStrategy
                    onCopyClicked: appState.copySelectedStrategy()
                }
            }

            LogConsole {
                id: logs
                Layout.fillWidth: true
                Layout.preferredHeight: expanded ? 210 : 42
                Layout.maximumHeight: expanded ? 260 : 42
                entries: appState.logs
            }
        }
    }
}
