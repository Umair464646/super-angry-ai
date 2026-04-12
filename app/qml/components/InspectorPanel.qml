import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    property var strategy: ({})
    signal copyClicked()

    color: "#0E1420"
    border.color: "#1A2638"
    radius: 12

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 10

        Label { text: "Inspector"; color: "#E0EDFF"; font.bold: true; font.pixelSize: 16 }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 10
            color: "#101B2C"
            border.color: "#1A2D47"

            ScrollView {
                anchors.fill: parent
                anchors.margins: 10
                TextArea {
                    text: strategy && strategy.id ?
                        ("Strategy ID: " + strategy.id + "\n" +
                        "Name: " + strategy.name + "\n" +
                        "Family: " + strategy.family + "\n" +
                        "Generation: " + strategy.generation + "\n" +
                        "Origin: " + strategy.origin + "\n\n" +
                        "Entry:\n- " + strategy.entry + "\n\n" +
                        "Exit:\n- " + strategy.exit + "\n\n" +
                        "Fitness: " + strategy.fitness + "\n" +
                        "Robustness: " + strategy.robustness + "\n" +
                        "Validation: " + strategy.validation)
                        : "Select a strategy to inspect full details."
                    readOnly: true
                    wrapMode: Text.Wrap
                    color: "#D7E5FA"
                }
            }
        }

        Button {
            text: "Copy Strategy"
            Layout.fillWidth: true
            onClicked: root.copyClicked()
        }
    }
}
